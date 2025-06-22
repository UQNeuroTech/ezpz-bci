import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QTextEdit
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

class InfoPage(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Info Page")
        self.setGeometry(100, 100, 600, 500)

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Title label
        self.title_label = QLabel("Brain Flow Protocol and Licensing Agreements", self)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        # Image
        self.image_label = QLabel(self)
        pixmap = QPixmap("images/brainflow.png")  # Path to the image
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(True)  # Allow scaling to fit the label without distortion
        self.image_label.setFixedSize(300, 200)  # Set the size of the label

        # Information text
        self.info_text = QTextEdit(self)
        self.info_text.setReadOnly(True)  # Make the text box read-only
        self.info_text.setStyleSheet("font-size: 14px;")
        self.info_text.setText("""
Brain Flow Protocol:
---------------------
Brain Flow is a protocol designed to facilitate communication between brain-computer interfaces (BCIs) and external devices. It provides a standardized framework for data acquisition, processing, and transmission, ensuring compatibility across various hardware and software platforms.

Key Features:
- Cross-platform compatibility
- Support for multiple hardware devices
- Real-time data streaming and processing
- Open-source and community-driven development

Licensing Agreements:
----------------------
Brain Flow is licensed under the MIT License, which permits the use, modification, and distribution of the software for both commercial and non-commercial purposes. The license ensures that the protocol remains open and accessible to developers worldwide.

MIT License:
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
        """)

        # Add widgets to layout
        self.layout.addWidget(self.title_label)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.info_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InfoPage()
    window.show()
    sys.exit(app.exec_())
