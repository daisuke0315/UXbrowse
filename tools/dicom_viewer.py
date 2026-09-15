#!/usr/bin/env python3
"""Small, local-only DICOM series viewer built with PyQt6 and pydicom."""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pydicom
from PyQt6.QtCore import QEvent, QPoint, Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QDesktopServices, QImage, QInputDevice, QKeySequence, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QSlider,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from viewsync_link import ViewSyncLink, ViewSyncRegistry, build_viewer_url


def number(value, default: float) -> float:
    """Return the first numeric DICOM value, tolerating MultiValue fields."""
    try:
        if isinstance(value, (list, tuple)) or hasattr(value, "__getitem__") and not isinstance(value, (str, bytes)):
            value = value[0]
        return float(value)
    except (TypeError, ValueError, IndexError):
        return default


def slice_sort_key(dataset: pydicom.Dataset) -> tuple[int, float]:
    position = getattr(dataset, "ImagePositionPatient", None)
    orientation = getattr(dataset, "ImageOrientationPatient", None)
    if position is not None and orientation is not None and len(position) >= 3 and len(orientation) >= 6:
        row = np.asarray(orientation[:3], dtype=float)
        column = np.asarray(orientation[3:6], dtype=float)
        return (0, float(np.dot(np.asarray(position[:3], dtype=float), np.cross(row, column))))
    return (1, number(getattr(dataset, "InstanceNumber", 0), 0))


class ImageView(QLabel):
    def __init__(self, window: "DicomViewer", orientation: str) -> None:
        super().__init__(orientation.upper())
        self.window = window
        self.orientation = orientation
        self.image_shape = (1, 1)
        self.image_rect = (0.0, 0.0, 1.0, 1.0)
        self.pan = [0.0, 0.0]
        self._tracking_cross = False
        self._panning = False
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 260)
        self.setStyleSheet("background:#080b10;color:#8792a2;font-size:16px;")
        self.setMouseTracking(True)
        self._last: QPoint | None = None

    def wheelEvent(self, event) -> None:
        if not self.window.datasets:
            return
        self.window.active_orientation = self.orientation
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.12 if event.angleDelta().y() > 0 else 0.89
            self.window.zoom_active(factor, self, (event.position().x(), event.position().y()))
        elif (
            not event.pixelDelta().isNull()
            or event.device().type() == QInputDevice.DeviceType.TouchPad
        ):
            # A two-finger trackpad scroll is navigation of the image itself,
            # not a request to change slices. Qt usually supplies pixelDelta;
            # the explicit device check covers macOS events that only carry an
            # angle delta.
            delta = event.pixelDelta()
            if delta.isNull():
                self.pan[0] += event.angleDelta().x() / 8.0
                self.pan[1] += event.angleDelta().y() / 8.0
            else:
                self.pan[0] += delta.x()
                self.pan[1] += delta.y()
            self.window.render_image()
        else:
            step = 1 if event.angleDelta().y() < 0 else -1
            self.window.step_axis(self.orientation, step)
        event.accept()

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.NativeGesture and event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
            self.window.active_orientation = self.orientation
            # On macOS the native pinch position can be expressed relative to an
            # ancestor window. Convert its global position explicitly so views
            # away from the top-left (coronal/sagittal) use the correct anchor.
            local = self.mapFromGlobal(event.globalPosition().toPoint())
            self.window.zoom_active(float(np.exp(event.value())), self, (local.x(), local.y()))
            event.accept()
            return True
        return super().event(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.window.active_orientation = self.orientation
            if self.window.is_near_cross(self.orientation, event.position().x(), event.position().y(), self):
                self._tracking_cross = True
                self.window.focus_from_view(self.orientation, event.position().x(), event.position().y(), self)
            else:
                self._panning = True
                self._last = event.position().toPoint()

    def mouseMoveEvent(self, event) -> None:
        if self._tracking_cross and event.buttons() & Qt.MouseButton.LeftButton:
            self.window.focus_from_view(self.orientation, event.position().x(), event.position().y(), self)
            return
        if self._panning and self._last is not None and event.buttons() & Qt.MouseButton.LeftButton:
            current = event.position().toPoint()
            delta = current - self._last
            self._last = current
            self.pan[0] += delta.x()
            self.pan[1] += delta.y()
            self.window.render_image()
            return
    def mouseReleaseEvent(self, event) -> None:
        self._last = None
        self._tracking_cross = False
        self._panning = False

    def mouseDoubleClickEvent(self, event) -> None:
        self.window.reset_view()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.window.render_image()

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        action = menu.addAction("ViewSyncで表示")
        action.setEnabled(self.window.view_sync_action.isEnabled())
        action.setToolTip(self.window.view_sync_action.toolTip())
        action.triggered.connect(self.window.open_view_sync)
        menu.exec(event.globalPos())


class DicomViewer(QMainWindow):
    def __init__(self, registry: ViewSyncRegistry, demo_dataset: str | None = None) -> None:
        super().__init__()
        self.registry = registry
        self.demo_dataset = demo_dataset
        self.current_view_sync_link: ViewSyncLink | None = None
        self.datasets: list[pydicom.Dataset] = []
        self.source_paths: list[Path] = []
        self.series_groups: dict[str, list[tuple[Path, pydicom.Dataset]]] = {}
        self.volume: np.ndarray | None = None
        self.focus = [0, 0, 0]  # x, y, z voxel indices
        self.current_pixels: np.ndarray | None = None
        self.pixel_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.pixel_cache_bytes = 0
        self.pixel_cache_budget = 256 * 1024 * 1024
        self.level = 40.0
        self.width = 400.0
        self.zooms = {"axial": 1.0, "coronal": 1.0, "sagittal": 1.0}
        self.active_orientation = "axial"
        self.inverted = False
        self._pending_tag_index = 0
        self.tag_timer = QTimer(self)
        self.tag_timer.setSingleShot(True)
        self.tag_timer.setInterval(250)
        self.tag_timer.timeout.connect(self._populate_pending_tags)
        self.setWindowTitle("PSP Mock Viewer")
        self.resize(1180, 780)
        self._build_ui()

    def _build_ui(self) -> None:
        toolbar = QToolBar("操作")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self._action(toolbar, "ファイルを開く", self.open_files, QKeySequence.StandardKey.Open)
        self._action(toolbar, "フォルダを開く", self.open_folder)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" シリーズ: "))
        self.series_combo = QComboBox()
        self.series_combo.setMinimumWidth(360)
        self.series_combo.setEnabled(False)
        self.series_combo.currentIndexChanged.connect(self.select_series)
        toolbar.addWidget(self.series_combo)
        toolbar.addSeparator()
        self._action(toolbar, "表示をリセット", self.reset_view, QKeySequence("R"))
        self._action(toolbar, "白黒反転", self.toggle_invert, QKeySequence("I"))
        toolbar.addSeparator()
        self._action(toolbar, "縮小 −", lambda: self.zoom_active(1 / 1.25), QKeySequence("-"))
        self._action(toolbar, "拡大 ＋", lambda: self.zoom_active(1.25), QKeySequence("+"))
        self._action(toolbar, "等倍", self.reset_active_zoom, QKeySequence("0"))
        toolbar.addSeparator()
        self.view_sync_action = self._action(toolbar, "ViewSyncで表示 ↗", self.open_view_sync)
        self.view_sync_action.setEnabled(False)
        self.view_sync_action.setToolTip("CTを読み込むと利用可否を確認します")

        self.views = {
            "axial": ImageView(self, "axial"),
            "coronal": ImageView(self, "coronal"),
            "sagittal": ImageView(self, "sagittal"),
        }
        self.image_view = self.views["axial"]
        self.sliders = {}
        for orientation in ("axial", "coronal", "sagittal"):
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setInvertedAppearance(False)
            slider.setEnabled(False)
            slider.valueChanged.connect(lambda value, name=orientation: self.move_orientation_slider(name, value))
            self.sliders[orientation] = slider
        self.slider = self.sliders["axial"]

        image_panel = QWidget()
        image_layout = QGridLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(3)
        for orientation, row, column in (
            ("axial", 0, 0),
            ("coronal", 0, 1),
            ("sagittal", 1, 0),
        ):
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            panel_layout.setSpacing(2)
            panel_layout.addWidget(self.views[orientation], 1)
            panel_layout.addWidget(self.sliders[orientation])
            image_layout.addWidget(panel, row, column)

        self.tags = QTableWidget(0, 3)
        self.tags.setHorizontalHeaderLabels(["Tag / VR", "項目", "値"])
        self.tags.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tags.setAlternatingRowColors(True)
        self.tags.verticalHeader().setVisible(False)
        header = self.tags.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.patient_label = QLabel("患者ID: —")
        self.study_label = QLabel("Study UID: —")
        self.view_sync_status_label = QLabel("ViewSync: CTを読み込んでください")
        self.view_sync_status_label.setStyleSheet("color:#7f8c8d;font-weight:600;")
        right_layout.addWidget(self.patient_label)
        right_layout.addWidget(self.study_label)
        right_layout.addWidget(self.view_sync_status_label)
        right_layout.addWidget(QLabel("十字座標ログ"))
        self.coordinate_log = QPlainTextEdit()
        self.coordinate_log.setReadOnly(True)
        self.coordinate_log.setMaximumBlockCount(1000)
        self.coordinate_log.setMaximumHeight(180)
        right_layout.addWidget(self.coordinate_log)
        right_layout.addWidget(QLabel("DICOMタグ"))
        right_layout.addWidget(self.tags, 1)

        splitter = QSplitter()
        splitter.addWidget(image_panel)
        splitter.addWidget(right)
        splitter.setSizes([820, 360])
        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("十字中心を左ドラッグ: 十字追従  その他を左ドラッグ: パン  ホイール: 断面移動  ピンチ: ズーム")

    def _action(self, toolbar: QToolBar, text: str, callback, shortcut=None) -> QAction:
        action = QAction(text, self)
        action.triggered.connect(callback)
        if shortcut:
            action.setShortcut(shortcut)
        toolbar.addAction(action)
        return action

    def open_files(self) -> None:
        names, _ = QFileDialog.getOpenFileNames(self, "DICOMファイルを選択", "", "DICOM (*.dcm *.dicom);;すべて (*)")
        if names:
            self.load_paths([Path(name) for name in names])

    def open_folder(self) -> None:
        name = QFileDialog.getExistingDirectory(self, "DICOMフォルダを選択")
        if name:
            self.load_paths([path for path in Path(name).rglob("*") if path.is_file()])

    def load_paths(self, paths: list[Path]) -> None:
        loaded: list[tuple[Path, pydicom.Dataset]] = []
        for path in paths:
            try:
                # Scan headers only. Pixel data can be hundreds of MB for a large
                # study and is loaded lazily only for the selected series/slice.
                dataset = pydicom.dcmread(path, force=False, stop_before_pixels=True)
                if hasattr(dataset, "Rows") and hasattr(dataset, "Columns"):
                    loaded.append((path, dataset))
            except Exception:
                continue
        if not loaded:
            QMessageBox.warning(self, "読込できません", "画像を含むDICOMファイルが見つかりませんでした。")
            return

        groups: dict[str, list[tuple[Path, pydicom.Dataset]]] = {}
        for item in loaded:
            uid = str(getattr(item[1], "SeriesInstanceUID", "unknown"))
            groups.setdefault(uid, []).append(item)
        for group in groups.values():
            group.sort(key=lambda item: slice_sort_key(item[1]))
        self.series_groups = groups

        ordered = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
        self.series_combo.blockSignals(True)
        self.series_combo.clear()
        for uid, group in ordered:
            header = group[0][1]
            description = str(getattr(header, "SeriesDescription", "名称なし")) or "名称なし"
            modality = str(getattr(header, "Modality", ""))
            number_value = str(getattr(header, "SeriesNumber", "-"))
            self.series_combo.addItem(
                f"{number_value} · {description} · {modality} · {len(group)}枚",
                uid,
            )
        self.series_combo.blockSignals(False)
        self.series_combo.setEnabled(len(ordered) > 1)
        self.series_combo.setCurrentIndex(0)
        self.select_series(0)
        self.statusBar().showMessage(f"{len(groups)}シリーズ、合計{len(loaded)}枚を検出しました")

    def select_series(self, combo_index: int) -> None:
        if combo_index < 0 or not self.series_groups:
            return
        uid = self.series_combo.itemData(combo_index)
        selected = self.series_groups.get(uid, [])
        if not selected:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            datasets: list[pydicom.Dataset] = []
            source_paths: list[Path] = []
            for path, _header in selected:
                try:
                    # Keep large Pixel Data deferred; pydicom reads it on demand.
                    datasets.append(pydicom.dcmread(path, force=False, defer_size="512 KB"))
                    source_paths.append(path)
                except Exception:
                    continue
            if not datasets:
                QMessageBox.warning(self, "読込できません", "選択したシリーズの画像を読み込めませんでした。")
                return
            self.source_paths = source_paths
            self.datasets = datasets
        finally:
            QApplication.restoreOverrideCursor()

        self.update_view_sync_state()

        self.current_pixels = None
        if not self.build_volume():
            return
        self.pixel_cache.clear()
        self.pixel_cache_bytes = 0
        depth, rows, columns = self.volume.shape
        self.focus = [columns // 2, rows // 2, depth // 2]
        for orientation, maximum in (
            ("axial", depth - 1),
            ("coronal", rows - 1),
            ("sagittal", columns - 1),
        ):
            slider = self.sliders[orientation]
            slider.blockSignals(True)
            slider.setRange(0, maximum)
            slider.setEnabled(maximum > 0)
            slider.blockSignals(False)
        self.update_orientation_sliders()
        for view in self.views.values():
            view.pan = [0.0, 0.0]
        self._set_initial_window(self.slider.value())
        self.show_slice(self.slider.value())
        self.log_focus("series")
        self.statusBar().showMessage(f"シリーズを読み込みました：{len(self.datasets)}枚")

    def update_view_sync_state(self) -> None:
        if not self.datasets:
            self.current_view_sync_link = None
            self.view_sync_action.setEnabled(False)
            return
        header = self.datasets[0]
        patient_id = str(getattr(header, "PatientID", "—")) or "—"
        study_uid = str(getattr(header, "StudyInstanceUID", ""))
        self.patient_label.setText(f"患者ID: {patient_id}")
        self.study_label.setText(f"Study UID: {study_uid or '—'}")
        link = self.registry.find(study_uid)
        if link is None and self.demo_dataset:
            link = ViewSyncLink(study_uid or "psp-mock-study", self.demo_dataset, "開発用デモ指定")
        self.current_view_sync_link = link
        available = link is not None
        self.view_sync_action.setEnabled(available)
        if available:
            self.view_sync_action.setToolTip("表示中の検査をViewSyncで開きます")
            self.view_sync_status_label.setText(f"ViewSync: 利用可能 · {link.label}")
            self.view_sync_status_label.setStyleSheet("color:#0f9d8a;font-weight:700;")
        else:
            reason = "この検査ではViewSync用画像が取得されていません"
            self.view_sync_action.setToolTip(reason)
            self.view_sync_status_label.setText(f"ViewSync: 利用不可 · {reason}")
            self.view_sync_status_label.setStyleSheet("color:#7f8c8d;font-weight:600;")

    def open_view_sync(self) -> None:
        link = self.current_view_sync_link
        if link is None:
            return
        url = build_viewer_url(self.registry.viewer_url, link)
        if not QDesktopServices.openUrl(QUrl(url)):
            QMessageBox.warning(self, "ViewSyncを開けません", f"既定ブラウザでURLを開けませんでした。\n\n{url}")
            return
        self.statusBar().showMessage("ViewSyncを外部ブラウザで開きました", 5000)

    def build_volume(self) -> bool:
        """Decode the selected series into a volume required by linked MPR views."""
        if not self.datasets:
            return False
        rows = int(getattr(self.datasets[0], "Rows", 0))
        columns = int(getattr(self.datasets[0], "Columns", 0))
        if not rows or not columns:
            return False
        estimated = len(self.datasets) * rows * columns * 4
        if estimated > 2 * 1024**3:
            answer = QMessageBox.question(
                self,
                "大きなシリーズ",
                f"3断面表示には約{estimated / 1024**3:.1f} GBのメモリが必要です。続けますか？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        progress = QProgressDialog("3断面ボリュームを構築しています…", "キャンセル", 0, len(self.datasets), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(250)
        volume = np.empty((len(self.datasets), rows, columns), dtype=np.float32)
        try:
            for index, dataset in enumerate(self.datasets):
                pixels = self.hu_pixels(dataset)
                if pixels.shape != (rows, columns):
                    raise ValueError("シリーズ内で画像サイズが一致していません")
                volume[index] = pixels
                progress.setValue(index + 1)
                if index % 8 == 0:
                    QApplication.processEvents()
                if progress.wasCanceled():
                    return False
        except Exception as exc:
            QMessageBox.critical(self, "MPRを構築できません", str(exc))
            return False
        finally:
            progress.close()
        self.volume = volume
        return True

    def _set_initial_window(self, index: int) -> None:
        dataset = self.datasets[index]
        pixels = self.get_pixels(index)
        self.level = number(getattr(dataset, "WindowCenter", None), float(np.median(pixels)))
        default_width = float(np.percentile(pixels, 99) - np.percentile(pixels, 1))
        self.width = max(1.0, number(getattr(dataset, "WindowWidth", None), default_width))
        self.zooms = {"axial": 1.0, "coronal": 1.0, "sagittal": 1.0}
        self.inverted = str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1"

    @staticmethod
    def hu_pixels(dataset: pydicom.Dataset) -> np.ndarray:
        pixels = dataset.pixel_array.astype(np.float32)
        slope = number(getattr(dataset, "RescaleSlope", 1), 1)
        intercept = number(getattr(dataset, "RescaleIntercept", 0), 0)
        return pixels * slope + intercept

    def get_pixels(self, index: int) -> np.ndarray:
        """Decode each visited slice once and retain a bounded LRU cache."""
        if self.volume is not None:
            return self.volume[index]
        cached = self.pixel_cache.pop(index, None)
        if cached is not None:
            self.pixel_cache[index] = cached
            return cached
        pixels = self.hu_pixels(self.datasets[index])
        self.pixel_cache[index] = pixels
        self.pixel_cache_bytes += pixels.nbytes
        while self.pixel_cache_bytes > self.pixel_cache_budget and len(self.pixel_cache) > 1:
            _old_index, old_pixels = self.pixel_cache.popitem(last=False)
            self.pixel_cache_bytes -= old_pixels.nbytes
        return pixels

    def show_slice(self, index: int) -> None:
        if not self.datasets:
            return
        self.focus[2] = index
        try:
            self.current_pixels = self.get_pixels(index)
        except Exception as exc:
            QMessageBox.critical(self, "画像を展開できません", f"圧縮形式用デコーダが必要な可能性があります。\n\n{exc}")
            return
        # Rebuilding a large tag table on every wheel tick is expensive. Update it
        # only after navigation has paused briefly.
        self._pending_tag_index = index
        self.tag_timer.start()
        self.render_image()
        self.log_focus("axial")

    def move_orientation_slider(self, orientation: str, value: int) -> None:
        x, y, z = self.focus
        if orientation == "axial":
            z = value
        elif orientation == "coronal":
            y = value
        else:
            x = value
        self.set_focus(x, y, z, orientation)

    def update_orientation_sliders(self) -> None:
        for orientation, value in zip(("sagittal", "coronal", "axial"), self.focus):
            slider = self.sliders[orientation]
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)

    def render_image(self) -> None:
        if self.volume is None or not hasattr(self, "views"):
            return
        x, y, z = self.focus
        images = {
            "axial": self.volume[z, :, :],
            "coronal": self.volume[:, y, :],
            "sagittal": self.volume[:, :, x],
        }
        crosses = {
            "axial": (x, y),
            "coronal": (x, z),
            "sagittal": (y, z),
        }
        for orientation, pixels in images.items():
            self.render_view(self.views[orientation], pixels, crosses[orientation])
        index = z + 1
        self.setWindowTitle(f"PSP Mock Viewer — {index}/{len(self.datasets)} — WL {self.level:.0f} / WW {self.width:.0f}")

    def render_view(self, view: ImageView, pixels: np.ndarray, cross: tuple[int, int]) -> None:
        low = self.level - self.width / 2.0
        image = np.clip((pixels - low) / self.width, 0.0, 1.0)
        if self.inverted:
            image = 1.0 - image
        image8 = np.ascontiguousarray((image * 255).astype(np.uint8))
        height, width = image8.shape[:2]
        view.image_shape = (height, width)
        qimage = QImage(image8.data, width, height, image8.strides[0], QImage.Format.Format_Grayscale8).copy()
        zoom = self.zooms[view.orientation]
        target_width = max(1, round(view.width() * zoom))
        target_height = max(1, round(view.height() * zoom))
        image_pixmap = QPixmap.fromImage(qimage).scaled(
            target_width,
            target_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        canvas = QPixmap(max(1, view.width()), max(1, view.height()))
        canvas.fill(Qt.GlobalColor.black)
        draw_x = (canvas.width() - image_pixmap.width()) / 2.0 + view.pan[0]
        draw_y = (canvas.height() - image_pixmap.height()) / 2.0 + view.pan[1]
        view.image_rect = (draw_x, draw_y, float(image_pixmap.width()), float(image_pixmap.height()))
        painter = QPainter(canvas)
        painter.drawPixmap(round(draw_x), round(draw_y), image_pixmap)
        painter.setPen(QPen(Qt.GlobalColor.red, 1))
        cross_x = draw_x + cross[0] / max(1, width - 1) * max(1, image_pixmap.width() - 1)
        cross_y = draw_y + cross[1] / max(1, height - 1) * max(1, image_pixmap.height() - 1)
        painter.drawLine(round(cross_x), round(draw_y), round(cross_x), round(draw_y + image_pixmap.height()))
        painter.drawLine(round(draw_x), round(cross_y), round(draw_x + image_pixmap.width()), round(cross_y))
        painter.setPen(QPen(Qt.GlobalColor.yellow, 1))
        painter.drawText(8, 18, view.orientation.upper())
        painter.end()
        view.setPixmap(canvas)

    def focus_from_view(self, orientation: str, mouse_x: float, mouse_y: float, view: ImageView) -> None:
        if self.volume is None:
            return
        offset_x, offset_y, image_width, image_height = view.image_rect
        px = (mouse_x - offset_x) / max(1, image_width)
        py = (mouse_y - offset_y) / max(1, image_height)
        if not (0 <= px <= 1 and 0 <= py <= 1):
            return
        height, width = view.image_shape
        column = int(np.clip(round(px * (width - 1)), 0, width - 1))
        row = int(np.clip(round(py * (height - 1)), 0, height - 1))
        x, y, z = self.focus
        if orientation == "axial":
            x, y = column, row
        elif orientation == "coronal":
            x, z = column, row
        else:
            y, z = column, row
        self.set_focus(x, y, z, orientation)

    def is_near_cross(self, orientation: str, mouse_x: float, mouse_y: float, view: ImageView) -> bool:
        if self.volume is None:
            return False
        offset_x, offset_y, image_width, image_height = view.image_rect
        height, width = view.image_shape
        x, y, z = self.focus
        if orientation == "axial":
            column, row = x, y
        elif orientation == "coronal":
            column, row = x, z
        else:
            column, row = y, z
        cross_x = offset_x + column / max(1, width - 1) * max(1, image_width - 1)
        cross_y = offset_y + row / max(1, height - 1) * max(1, image_height - 1)
        return abs(mouse_x - cross_x) <= 14 and abs(mouse_y - cross_y) <= 14

    def set_focus(self, x: int, y: int, z: int, source: str = "focus") -> None:
        if self.volume is None:
            return
        depth, rows, columns = self.volume.shape
        next_focus = [
            int(np.clip(x, 0, columns - 1)),
            int(np.clip(y, 0, rows - 1)),
            int(np.clip(z, 0, depth - 1)),
        ]
        if next_focus == self.focus:
            return
        self.focus = next_focus
        self.update_orientation_sliders()
        self.current_pixels = self.volume[self.focus[2]]
        self._pending_tag_index = self.focus[2]
        self.tag_timer.start()
        self.render_image()
        self.log_focus(source)

    def step_axis(self, orientation: str, step: int) -> None:
        x, y, z = self.focus
        if orientation == "axial":
            z += step
        elif orientation == "coronal":
            y += step
        else:
            x += step
        self.set_focus(x, y, z, orientation)

    def zoom_active(self, factor: float, view: ImageView | None = None, anchor: tuple[float, float] | None = None) -> None:
        current = self.zooms[self.active_orientation]
        updated = float(np.clip(current * factor, 0.25, 8.0))
        if view is not None and anchor is not None and updated != current:
            old_x, old_y, old_width, old_height = view.image_rect
            relative_x = (anchor[0] - old_x) / max(1.0, old_width)
            relative_y = (anchor[1] - old_y) / max(1.0, old_height)
        self.zooms[self.active_orientation] = updated
        self.render_image()
        if view is not None and anchor is not None and updated != current:
            # Use the actual post-scale integer pixmap dimensions. This avoids
            # cumulative anchor drift on narrow coronal/sagittal images during
            # the many tiny scale increments emitted by a trackpad pinch.
            new_x, new_y, new_width, new_height = view.image_rect
            actual_x = new_x + relative_x * new_width
            actual_y = new_y + relative_y * new_height
            view.pan[0] += anchor[0] - actual_x
            view.pan[1] += anchor[1] - actual_y
            self.render_image()

    def reset_active_zoom(self) -> None:
        self.zooms[self.active_orientation] = 1.0
        self.views[self.active_orientation].pan = [0.0, 0.0]
        self.render_image()

    def patient_coordinate(self) -> tuple[float, float, float] | None:
        if not self.datasets:
            return None
        x, y, z = self.focus
        dataset = self.datasets[z]
        position = getattr(dataset, "ImagePositionPatient", None)
        orientation = getattr(dataset, "ImageOrientationPatient", None)
        spacing = getattr(dataset, "PixelSpacing", None)
        if position is None or orientation is None or spacing is None:
            return None
        origin = np.asarray(position[:3], dtype=float)
        row_direction = np.asarray(orientation[:3], dtype=float)
        column_direction = np.asarray(orientation[3:6], dtype=float)
        point = origin + row_direction * x * float(spacing[1]) + column_direction * y * float(spacing[0])
        return float(point[0]), float(point[1]), float(point[2])

    def log_focus(self, source: str) -> None:
        x, y, z = self.focus
        patient = self.patient_coordinate()
        message = f"{source}: voxel (x={x}, y={y}, z={z})"
        if patient is not None:
            message += f"  patient mm (X={patient[0]:.2f}, Y={patient[1]:.2f}, Z={patient[2]:.2f})"
        self.coordinate_log.appendPlainText(message)
        self.coordinate_log.ensureCursorVisible()

    def populate_tags(self, dataset: pydicom.Dataset) -> None:
        elements = [element for element in dataset.iterall() if element.tag.group != 0x7FE0]
        self.tags.setUpdatesEnabled(False)
        self.tags.clearContents()
        self.tags.setRowCount(len(elements))
        for row, element in enumerate(elements):
            value = str(element.value)
            if len(value) > 500:
                value = value[:500] + "…"
            self.tags.setItem(row, 0, QTableWidgetItem(f"{element.tag} {element.VR}"))
            self.tags.setItem(row, 1, QTableWidgetItem(element.name))
            self.tags.setItem(row, 2, QTableWidgetItem(value))
        self.tags.setUpdatesEnabled(True)

    def _populate_pending_tags(self) -> None:
        if self.datasets:
            self.populate_tags(self.datasets[self._pending_tag_index])

    def reset_view(self) -> None:
        if self.datasets:
            for view in self.views.values():
                view.pan = [0.0, 0.0]
            self._set_initial_window(self.slider.value())
            self.show_slice(self.slider.value())

    def toggle_invert(self) -> None:
        self.inverted = not self.inverted
        self.render_image()


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PSP連携フローを検証するDICOMビューアモック")
    parser.add_argument("dicom_folder", nargs="?", type=Path, help="PSPから渡されるDICOM保存フォルダ")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(__file__).with_name("viewsync_studies.json"),
        help="Study Instance UIDとViewSyncデータの対応表",
    )
    parser.add_argument("--viewsync-url", help="対応表のviewerUrlを一時的に上書き")
    parser.add_argument("--demo-dataset", choices=("mikan", "banana", "all"), help="開発時だけ可否判定を上書き")
    return parser


def main() -> int:
    args = argument_parser().parse_args()
    registry = ViewSyncRegistry.from_file(args.registry)
    if args.viewsync_url:
        registry.viewer_url = args.viewsync_url
    app = QApplication([sys.argv[0]])
    app.setApplicationName("PSP Mock Viewer")
    app.setStyle("Fusion")
    window = DicomViewer(registry, args.demo_dataset)
    window.show()
    if args.dicom_folder:
        if not args.dicom_folder.is_dir():
            QMessageBox.warning(window, "フォルダが見つかりません", str(args.dicom_folder))
        else:
            paths = [path for path in args.dicom_folder.rglob("*") if path.is_file()]
            QTimer.singleShot(0, lambda: window.load_paths(paths))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
