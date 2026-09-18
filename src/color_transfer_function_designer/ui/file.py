from pathlib import Path

from trame.decorators import TrameApp, change
from trame.widgets import html
from trame.widgets import vuetify3 as v3

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

DIRECTORY = {"icon": "mdi-folder", "type": "directory"}
GROUP = {"icon": "mdi-file-document-multiple-outline", "type": "group"}
FILE = {"icon": "mdi-file-document-outline", "type": "file"}

HEADERS = [
    {"title": "Name", "align": "start", "key": "name", "sortable": False},
    {"title": "Size", "align": "end", "key": "size", "sortable": False},
    {"title": "Date", "align": "end", "key": "modified", "sortable": False},
]


def sort_by_name(e):
    return e.get("name")


def to_type(e):
    return e.get("type", "")


def to_suffix(e):
    return Path(e.get("name", "")).suffix


# -----------------------------------------------------------------------------


class FileBrowser:
    def __init__(self, home=None, current=None):
        self._enable_groups = True
        self._home_path = Path(home).resolve() if home else Path.home()
        self._current_path = Path(current).resolve() if current else self._home_path

    @property
    def enable_groups(self):
        return self._enable_groups

    @enable_groups.setter
    def enable_groups(self, v):
        self._enable_groups = v

    @property
    def listing(self):
        directories = []
        files = []
        for f in self._current_path.iterdir():
            if f.name[0] == ".":
                continue
            entry = {"name": f.name, "modified": f.stat().st_mtime}
            if f.is_dir():
                directories.append({**entry, **DIRECTORY})
            elif f.is_file():
                files.append({**entry, **FILE, "size": f.stat().st_size})

        # Sort content
        directories.sort(key=sort_by_name)
        files.sort(key=sort_by_name)

        return [{**e, "index": i} for i, e in enumerate([*directories, *files])]

    def open_entry(self, entry):
        entry_type = entry.get("type")
        if entry_type in ["directory", "file"]:
            self._current_path = self._current_path / entry.get("name")
            return entry_type, str(self._current_path)
        if entry_type == "group":
            files = entry.get("files", [])
            return entry, [str(self._current_path / f) for f in files]
        return None

    @property
    def current_path(self):
        return self._current_path

    def goto_home(self):
        self._current_path = self._home_path

    def goto_parent(self):
        self._current_path = self._current_path.parent

    def open_file(self, entry):
        return self._current_path / entry.get("name")


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# GUI Components
# -----------------------------------------------------------------------------


@TrameApp()
class FileDialog(v3.VDialog):
    def __init__(
        self,
        file_browser=None,
        is_open=True,
        **_,
    ):
        super().__init__(v_model=("file_dialog_is_open", is_open))

        # Initialize file browser
        if file_browser is None:
            file_browser = FileBrowser()
        self._file_browser = file_browser

        # Fill content for home
        self.goto_home()
        self.selected_entry = None

        # Define UI
        with self, v3.VCard(classes="mx-10"):
            style_align_center = "d-flex align-center "
            v3.VCardTitle(
                "Save File",
                v_show=("file_dialog_save_mode"),
                classes="text-center bg-grey-lighten-2",
            )
            v3.VCardTitle(
                "Open File",
                v_show=("!file_dialog_save_mode"),
                classes="text-center bg-grey-lighten-2",
            )
            with v3.VToolbar(density="compact", color="white"):
                v3.VBtn(
                    icon="mdi-home",
                    variant="flat",
                    size="small",
                    click=self.goto_home,
                )
                v3.VBtn(
                    icon="mdi-folder-upload-outline",
                    variant="flat",
                    size="small",
                    click=self.goto_parent,
                )
                v3.VTextField(
                    v_model=("file_dialog_filter", ""),
                    hide_details=True,
                    color="primary",
                    placeholder="filter",
                    density="compact",
                    variant="outlined",
                    classes="mx-2",
                    prepend_inner_icon="mdi-magnify",
                    clearable=True,
                )
            with v3.VDataTable(
                density="compact",
                fixed_header=True,
                headers=("file_dialog_headers", HEADERS),
                items=("file_dialog_listing", []),
                height="50vh",
                style="user-select: none; cursor: pointer;",
                hover=True,
                search=("file_dialog_filter",),
                items_per_page=-1,
            ):
                v3.Template(raw_attrs=["v-slot:bottom"])
                with v3.Template(raw_attrs=['v-slot:item="{ index, item }"']):
                    with v3.VDataTableRow(
                        index=("index",),
                        item=("item",),
                        click=(self.select_entry, "[item]"),
                        dblclick=(self.open_entry, "[item]"),
                        classes=(
                            "{ 'bg-grey': item.index === file_dialog_active, 'cursor-pointer': 1 }",
                        ),
                    ):
                        with v3.Template(raw_attrs=["v-slot:item.name"]):
                            with html.Div(classes=style_align_center):
                                v3.VIcon(
                                    "{{ item.icon }}",
                                    size="small",
                                    classes="mr-2",
                                )
                                html.Div("{{ item.name }}")

                        with v3.Template(raw_attrs=["v-slot:item.size"]):
                            with html.Div(
                                classes=style_align_center + " justify-end",
                            ):
                                html.Div(
                                    "{{ utils.fmt.bytes(item.size, 0) }}",
                                    v_if="item.size",
                                )
                                html.Div(" - ", v_else=True)

                        with v3.Template(raw_attrs=["v-slot:item.modified"]):
                            with html.Div(
                                classes=style_align_center + " justify-end",
                            ):
                                html.Div(
                                    "{{ new Date(item.modified * 1000).toDateString() }}"
                                )

            v3.VTextField(
                v_show=("file_dialog_save_mode", False),
                v_model=("file_dialog_save_filename", ""),
                label="File name",
                placeholder="output.vp",
                density="compact",
                variant="outlined",
                classes="mx-4 my-2",
                hide_details=True,
            )
            with v3.VCardActions():
                v3.VCheckbox(
                    v_show=("!file_dialog_save_mode",),
                    v_model=("file_dialog_groups", True),
                    density="compact",
                    hide_details=True,
                    false_icon="mdi-file-document-outline",
                    true_icon="mdi-file-document-multiple-outline",
                    label=("` Groups ${file_dialog_groups ? 'enabled' : 'disabled'}`",),
                    classes="mx-2",
                )
                v3.VSpacer()
                v3.VBtn(
                    text="Save",
                    v_show=("file_dialog_save_mode",),
                    variant="tonal",
                    disabled=("!file_dialog_save_filename",),
                    click=(
                        self.confirm,
                        "[file_dialog_listing[file_dialog_active]]",
                    ),
                )
                v3.VBtn(
                    text="Open",
                    v_show=("!file_dialog_save_mode",),
                    variant="tonal",
                    disabled=("file_dialog_open_disabled", True),
                    click=(
                        self.confirm,
                        "[file_dialog_listing[file_dialog_active]]",
                    ),
                )
                v3.VBtn("Cancel", variant="tonal", click="file_dialog_is_open = false")

    @property
    def state(self):
        return self.server.state

    @property
    def controller(self):
        return self.server.controller

    def _update_listing(self):
        self.selected_entry = None
        self.state.file_dialog_active = -1
        self.state.file_dialog_listing = self._file_browser.listing

    def goto_home(self):
        self._file_browser.goto_home()
        self._update_listing()

    def goto_parent(self):
        self._file_browser.goto_parent()
        self._update_listing()

    def select_entry(self, entry):
        self.selected_entry = entry
        self.state.file_dialog_active = entry.get("index", 0) if entry else -1
        self.state.file_dialog_open_disabled = True
        if entry and to_type(entry) in ["file", "group"]:
            self.state.file_dialog_open_disabled = False
            if self.state.file_dialog_save_mode:
                self.state.file_dialog_save_filename = entry.get("name", "")

    def open_entry(self, entry):
        if to_type(entry) == "directory":
            self._file_browser.open_entry(entry)
            self._update_listing()
        elif self.state.file_dialog_save_mode:
            self.state.file_dialog_save_filename = entry.get("name", "")
        else:
            self.open_file(entry)

    def confirm(self, entry=None):
        if self.state.file_dialog_save_mode:
            self._save_file()
        else:
            self.open_file(entry)

    def open_file(self, entry):
        self.state.file_dialog_is_open = False
        event = self._file_browser.open_file(entry)
        if self.controller.on_file_open.exists():
            self.controller.on_file_open(event)

    def _save_file(self):
        name = self.state.file_dialog_save_filename.strip()
        if not name:
            return
        if not name.endswith(".vp"):
            name += ".vp"
        path = self._file_browser.current_path / name
        self.state.file_dialog_is_open = False
        self.state.file_dialog_save_mode = False
        if self.controller.on_file_save.exists():
            self.controller.on_file_save(path)

    @change("file_dialog_groups")
    def on_enable_group_change(self, file_dialog_groups, **_):
        self._file_browser.enable_groups = file_dialog_groups
        self._update_listing()


class OpenFileToggle(v3.VBtn):
    def __init__(self, **kwargs):
        super().__init__(
            icon="mdi-file-document-plus-outline",
            click="file_dialog_is_open = !file_dialog_is_open",
            **kwargs,
        )
        self.dialog = FileDialog(is_open=False)
