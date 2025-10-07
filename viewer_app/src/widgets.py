from PySide6 import QtCore
import pyqtgraph as pg
from typing import List, Dict, Optional, Tuple

class ClickableRegion(pg.LinearRegionItem):
    """
    A clickable region on a PyQtGraph plot that emits a signal when clicked.

    This subclass of LinearRegionItem is used to represent a labeled region
    (for example, an audio segment) that can respond to mouse clicks.
    """

    # Define a signal that emits the region instance when clicked
    clicked = QtCore.Signal(object)

    def __init__(self, *args, seg_id: Optional[str] = None, **kwargs):
        """
        Initialize a non-movable clickable region.
        """
        super().__init__(*args, **kwargs)

        # Store optional segment ID
        self.seg_id = seg_id

        # Disable dragging; region is static (clickable only)
        self.setMovable(False)

    def mouseClickEvent(self, ev):
        """
        Handle mouse click events on the region.

        Emits the `clicked` signal when the left mouse button is pressed.
        """
        # Check if the left mouse button was clicked
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            # Emit the clicked signal with this region as the argument
            self.clicked.emit(self)

            # Accept the event to stop further propagation
            ev.accept()
        else:
            # Ignore other mouse buttons
            ev.ignore()