import sys

from PySide6 import QtWidgets

from src.app_window import App


def main() -> None:
    qt_app = QtWidgets.QApplication(sys.argv)

    window = App()
    window.show()

    sys.exit(qt_app.exec())


if __name__ == "__main__":
    main()