# -*- coding: utf-8 -*-
"""PyQt6 GUI for the DICOM gap-filling preprocessor."""

import contextlib
import os
import sys
import traceback
from pathlib import Path

from PyQt6.QtCore import QObject, QSettings, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from .ipp import run_preprocess, scan_series
except ImportError:
    from ipp import run_preprocess, scan_series


class StreamEmitter(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        if text:
            self.text_written.emit(str(text))

    def flush(self):
        pass


class PreprocessWorker(QObject):
    log = pyqtSignal(str)
    completed = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, input_dir: str, output_dir: str, series_uid: str):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.series_uid = series_uid

    def run(self):
        stream = StreamEmitter()
        stream.text_written.connect(self.log)
        try:
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                report = run_preprocess(self.input_dir, self.output_dir, self.series_uid)
            self.completed.emit(report)
        except Exception:
            self.failed.emit(traceback.format_exc())


class ScanWorker(QObject):
    completed = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, input_dir: str):
        super().__init__()
        self.input_dir = input_dir

    def run(self):
        try:
            self.completed.emit(scan_series(self.input_dir))
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("UXbrowse", "DicomGapFiller")
        self.thread = None
        self.worker = None
        self.scanned_input_dir = ""
        self.setWindowTitle("DICOM 欠損スライス補完")
        self.resize(980, 700)
        self._build_ui()
        self._restore_paths()

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)

        intro = QLabel(
            "DICOM CTシリーズを物理位置順に並べ、抜けたZ断面を前後スライスから線形補間します。\n"
            "元のDICOMファイルは変更せず、補完済みのNIfTI / MHAとレポートを別フォルダへ保存します。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        paths = QGroupBox("フォルダ")
        grid = QGridLayout(paths)
        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        input_button = QPushButton("選択…")
        self.scan_button = QPushButton("シリーズを確認")
        output_button = QPushButton("選択…")
        input_button.clicked.connect(self._choose_input)
        self.scan_button.clicked.connect(self._scan)
        output_button.clicked.connect(self._choose_output)
        grid.addWidget(QLabel("DICOM入力"), 0, 0)
        grid.addWidget(self.input_edit, 0, 1)
        grid.addWidget(input_button, 0, 2)
        grid.addWidget(self.scan_button, 0, 3)
        grid.addWidget(QLabel("出力先"), 1, 0)
        grid.addWidget(self.output_edit, 1, 1)
        grid.addWidget(output_button, 1, 2)
        layout.addWidget(paths)

        series_group = QGroupBox("検出されたシリーズ（補完するシリーズを1つ選択）")
        series_layout = QVBoxLayout(series_group)
        self.series_table = QTableWidget(0, 6)
        self.series_table.setHorizontalHeaderLabels(
            ["説明", "番号", "モダリティ", "撮影日時", "枚数", "SeriesInstanceUID"]
        )
        self.series_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.series_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.series_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.series_table.horizontalHeader().setStretchLastSection(True)
        series_layout.addWidget(self.series_table)
        layout.addWidget(series_group, 1)

        actions = QHBoxLayout()
        self.run_button = QPushButton("補完を実行")
        self.open_button = QPushButton("出力フォルダを開く")
        self.open_button.setEnabled(False)
        self.run_button.clicked.connect(self._start)
        self.open_button.clicked.connect(self._open_output)
        actions.addWidget(self.run_button)
        actions.addWidget(self.open_button)
        actions.addStretch()
        layout.addLayout(actions)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        layout.addWidget(QLabel("処理ログ"))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, 1)
        self.setCentralWidget(root)

    def _restore_paths(self):
        self.input_edit.setText(self.settings.value("input_dir", ""))
        self.output_edit.setText(self.settings.value("output_dir", ""))

    def _choose_input(self):
        path = QFileDialog.getExistingDirectory(self, "DICOM入力フォルダを選択", self.input_edit.text())
        if path:
            self.input_edit.setText(path)
            self._clear_series()
            if not self.output_edit.text():
                self.output_edit.setText(str(Path(path).parent / f"{Path(path).name}_preprocessed"))
            self._scan()

    def _choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "出力フォルダを選択", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _validate(self):
        input_dir = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        if not os.path.isdir(input_dir):
            raise ValueError("有効なDICOM入力フォルダを選択してください。")
        if not output_dir:
            raise ValueError("出力フォルダを指定してください。")
        if os.path.abspath(input_dir) == os.path.abspath(output_dir):
            raise ValueError("出力先は入力フォルダとは別にしてください。")
        if self.scanned_input_dir != os.path.abspath(input_dir):
            raise ValueError("入力フォルダのシリーズを確認してください。")
        row = self.series_table.currentRow()
        if row < 0:
            raise ValueError("補完するシリーズを一覧から選択してください。")
        series_uid = self.series_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        return input_dir, output_dir, series_uid

    def _clear_series(self):
        self.series_table.setRowCount(0)
        self.scanned_input_dir = ""

    def _scan(self):
        input_dir = self.input_edit.text().strip()
        if not os.path.isdir(input_dir):
            QMessageBox.warning(self, "入力を確認してください", "有効なDICOM入力フォルダを選択してください。")
            return
        self._clear_series()
        self.log_view.clear()
        self.log_view.appendPlainText("DICOMシリーズを確認しています…")
        self._set_busy(True)
        self.thread = QThread(self)
        self.worker = ScanWorker(input_dir)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.completed.connect(lambda series: self._scan_completed(input_dir, series))
        self.worker.failed.connect(self._failed)
        self.worker.completed.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()

    def _scan_completed(self, input_dir, series_list):
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        if not series_list:
            self.log_view.appendPlainText("読み取り可能なDICOMシリーズが見つかりませんでした。")
            return
        self.scanned_input_dir = os.path.abspath(input_dir)
        self.series_table.setRowCount(len(series_list))
        for row, series in enumerate(series_list):
            date = series["series_date"] or series["study_date"]
            time = str(series["series_time"]).split(".")[0]
            date_time = " ".join(part for part in (date, time) if part)
            values = [
                series["series_description"] or "（説明なし）",
                series["series_number"],
                series["modality"],
                date_time,
                str(series["slice_count"]),
                series["series_instance_uid"],
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, series["series_instance_uid"])
                self.series_table.setItem(row, column, item)
        self.series_table.resizeColumnsToContents()
        self.series_table.selectRow(0)
        self.log_view.appendPlainText(f"{len(series_list)}件のシリーズを検出しました。補完対象を選択してください。")

    def _set_busy(self, busy):
        self.run_button.setEnabled(not busy)
        self.scan_button.setEnabled(not busy)
        if busy:
            self.progress.setRange(0, 0)

    def _start(self):
        try:
            input_dir, output_dir, series_uid = self._validate()
        except ValueError as exc:
            QMessageBox.warning(self, "入力を確認してください", str(exc))
            return

        self.settings.setValue("input_dir", input_dir)
        self.settings.setValue("output_dir", output_dir)
        self.log_view.clear()
        self.log_view.appendPlainText("補完処理を開始します…")
        self._set_busy(True)
        self.open_button.setEnabled(False)

        self.thread = QThread(self)
        self.worker = PreprocessWorker(input_dir, output_dir, series_uid)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self._append_log)
        self.worker.completed.connect(self._completed)
        self.worker.failed.connect(self._failed)
        self.worker.completed.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()

    def _append_log(self, text):
        self.log_view.moveCursor(self.log_view.textCursor().MoveOperation.End)
        self.log_view.insertPlainText(text)
        self.log_view.ensureCursorVisible()

    def _completed(self, report):
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.open_button.setEnabled(True)
        message = (
            f"完了しました。\n\n"
            f"入力: {report['input_slice_count']}枚\n"
            f"出力: {report['output_slice_count']}枚\n"
            f"補完生成: {report['interpolated_slices_created']}枚"
        )
        QMessageBox.information(self, "補完完了", message)

    def _failed(self, details):
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.log_view.appendPlainText("\n[エラー]\n" + details)
        QMessageBox.critical(self, "処理に失敗しました", "ログのエラー内容を確認してください。")

    def _thread_finished(self):
        self._set_busy(False)
        if self.worker is not None:
            self.worker.deleteLater()
        if self.thread is not None:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None

    def _open_output(self):
        output_dir = self.output_edit.text().strip()
        if os.path.isdir(output_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("DICOM 欠損スライス補完")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
