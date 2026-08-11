from bimanual_collection.tools import find_cameras


def test_open_contact_sheet_uses_system_viewer_on_linux(tmp_path, monkeypatch):
    contact_sheet = tmp_path / "contact_sheet.jpg"
    contact_sheet.touch()
    calls = []

    class FakeProcess:
        def wait(self):
            return 0

    def fake_popen(command):
        calls.append(command)
        return FakeProcess()

    monkeypatch.setattr(find_cameras.platform, "system", lambda: "Linux")
    monkeypatch.setattr(find_cameras.subprocess, "Popen", fake_popen)

    assert find_cameras.open_contact_sheet_window(contact_sheet)
    assert calls == [["xdg-open", str(contact_sheet)]]


def test_open_contact_sheet_returns_false_for_missing_file(tmp_path):
    assert not find_cameras.open_contact_sheet_window(tmp_path / "missing.jpg")
