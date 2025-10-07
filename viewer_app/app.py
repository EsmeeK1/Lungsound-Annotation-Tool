from PySide6 import QtWidgets
from src.mainwindow import App

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())
