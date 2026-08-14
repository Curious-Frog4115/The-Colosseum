"""
Arena API + agent-runtime tests. Runs standalone (`python test_api.py`) or
under pytest. Uses a throwaway temp DB; the real arena.db is never touched.
"""
import io
import json
import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(__file__))
import main  # noqa: E402

_TMP = tempfile.mkdtemp(prefix="arena_test_")
main.DB_BASE = _TMP
main.GEN_DIR = _TMP
main.DB_PATH = os.path.join(_TMP, "arena.db")


async def _noop():
    pass


main.ensure_bazaar_key = _noop
main.init_db()
main.set_setting("admin_unlocked", "0")


def test_normalizer_removes_reasoning():
    n = main.StreamNormalizer()
    out = n.feed("Hello<think")
    out += n.feed("ing>secret chain of thought</thinking> world")
    out += n.flush()
    assert "secret" not in out
    assert "Hello" in out and "world" in out


def test_normalizer_captures_tool_xml_across_chunks():
    n = main.StreamNormalizer()
    out = n.feed("Before ")
    out += n.feed("<tool_call>\n<function=create_file>\n")
    out += n.feed("<parameter=name>\nindex.html\n</parameter>\n")
    out += n.feed("<parameter=content>\n<h1>hi</h1>\n</parameter>\n")
    out += n.feed("</tool_call>")
    out += n.flush()
    assert "tool_call" not in out and "create_file" not in out
    assert "Before" in out
    assert len(n.tool_blocks) == 1


def test_parse_xml_tool_block():
    block = ("<tool_call>\n<function=create_file>\n<parameter=name>index.html</parameter>\n"
             "<parameter=content>hello</parameter>\n</tool_call>")
    parsed = main.parse_xml_tool_block(block)
    assert parsed is not None
    tool, args = parsed
    assert tool == "create_file"
    assert args["name"] == "index.html"
    assert "hello" in args["content"]


def test_normalizer_captures_bare_json_tool_line():
    n = main.StreamNormalizer()
    n.feed('{"tool":"run_python","args":{"code":"print(1)"}}')
    out = n.flush()
    assert '"tool"' not in out
    assert n.tool_blocks and "run_python" in n.tool_blocks[0]


def test_calculate_tool():
    assert main.tool_calculate({"expression": "2 + 3 * 4"})["result"] == 14
    assert main.tool_calculate({"expression": "sqrt(16)"})["result"] == 4
    assert "error" in main.tool_calculate({"expression": "2 / 0"}) or \
           main.tool_calculate({"expression": "2/0"}).get("result") in (float("inf"),)


CONV = "testconv"


def test_canvas_tool_flow():
    r = main.tool_create_file({"name": "index.html", "content": "<html>hi</html>"}, CONV)
    assert r.get("created") == "index.html" and "error" not in r
    assert main.tool_create_file({"name": "index.html", "content": ""}, CONV)["error"]
    r = main.tool_read_file({"name": "index.html"}, CONV)
    assert "<html>hi</html>" in r["content"]
    assert "error" in main.tool_read_file({"name": "nope.js"}, CONV)

    r = main.tool_edit_file({"name": "index.html", "old": "<html>hi</html>",
                             "new": "<html>yo</html>"}, CONV)
    assert r.get("edited") == "index.html"
    assert "yo" in main.tool_read_file({"name": "index.html"}, CONV)["content"]
    assert "error" in main.tool_edit_file({"name": "index.html", "old": "missing"}, CONV)
    assert "error" in main.tool_edit_file({"name": "missing.html", "old": "x", "new": "y"}, CONV)

    r = main.tool_append_file({"name": "app.js", "content": "console.log(1);"}, CONV)
    assert r.get("appended") == "app.js"
    assert "error" in main.tool_append_file({"name": "app.js", "content": ""}, CONV)

    files = main.tool_list_files({}, CONV)
    assert files["count"] == 2
    names = {f["name"] for f in files["files"]}
    assert names == {"index.html", "app.js"}

    r = main.tool_delete_file({"name": "app.js"}, CONV)
    assert r.get("deleted") == "app.js"
    assert "error" in main.tool_delete_file({"name": "app.js"}, CONV)
    assert main.tool_list_files({}, CONV)["count"] == 1


def test_clean_final_strips_protocol():
    dirt = 'Some text\n{"tool":"create_file","args":{}}\n\nmore text'
    assert '"tool"' not in main.clean_final(dirt)
    assert main.clean_final("plain answer") == "plain answer"


_TC = None


def _client():
    global _TC
    if _TC is None:
        from fastapi.testclient import TestClient
        _TC = TestClient(main.app)
        _TC.__enter__()
    return _TC


def test_admin_unlock_flow():
    c = _client()
    assert c.get("/api/admin/status").json()["unlocked"] is False
    assert c.post("/api/admin/unlock", json={"password": "nope"}).status_code == 403
    ok = c.post("/api/admin/unlock", json={"password": "ai4freeadmin"})
    assert ok.status_code == 200 and ok.json()["unlocked"] is True
    d = c.get("/api/models").json()
    assert d["admin_unlocked"] is True
    frontier = next(m for m in d["models"] if m["category"] == "frontier")
    assert frontier["direct"] is True
    c.post("/api/admin/lock", json={"password": "ai4freeadmin"})
    assert c.get("/api/admin/status").json()["unlocked"] is False


def test_direct_chat_gate():
    c = _client()
    body = {"prompt": "hi", "model_id": "qwen35-397b", "conversation_id": ""}
    locked = c.post("/api/chat", json=body)
    assert locked.status_code == 403
    c.post("/api/admin/unlock", json={"password": "ai4freeadmin"})
    unlocked = c.post("/api/chat", json=body)
    assert unlocked.status_code == 200  # gate passes; stream not consumed here
    unlocked.close()
    c.post("/api/admin/lock", json={"password": "ai4freeadmin"})


def test_workspace_zip_and_download():
    c = _client()
    main.tool_create_file({"name": "index.html", "content": "<h1>zip</h1>"}, CONV)
    main.tool_create_file({"name": "style.css", "content": "body{}"}, CONV)
    r = c.get(f"/api/canvas/{CONV}")
    assert r.status_code == 200 and len(r.json()["files"]) >= 2

    z = c.get(f"/api/canvas/{CONV}/zip")
    assert z.status_code == 200
    assert z.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert "index.html" in names and "style.css" in names

    dl = c.get(f"/p/{CONV}/index.html", params={"dl": 1})
    assert dl.status_code == 200
    assert "attachment" in dl.headers["content-disposition"]
    assert "<h1>zip</h1>" in dl.text
    plain = c.get(f"/p/{CONV}/index.html")
    assert "attachment" not in plain.headers.get("content-disposition", "")


def _run_all():
    import re
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok  {name}")
        except Exception as e:
            fails += 1
            print(f"FAIL  {name}: {e!r}")
    print(f"\n{len(tests) - fails}/{len(tests)} tests passed")
    return fails


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)