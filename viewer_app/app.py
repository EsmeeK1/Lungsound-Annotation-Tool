from PySide6 import QtWidgets, QtCore
from src.mainwindow import App

def _excepthook(exc_type, exc, tb):
    import traceback
    traceback.print_exception(exc_type, exc, tb)
    msg = "".join(traceback.format_exception(exc_type, exc, tb))[-2000:]
    QtWidgets.QMessageBox.critical(None, "Onverwachte fout", msg)

def closeEvent(self, ev):
    print("[DEBUG] closeEvent fired — someone is trying to close the window")
    # Tijdelijk blokkeren om te zien of er programmatic een close volgt:
    ev.ignore()

if __name__ == "__main__":
    import sys
    sys.excepthook = _excepthook
    app = QtWidgets.QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())

