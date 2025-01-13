# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui_main.ui'
##
## Created by: Qt User Interface Compiler version 6.7.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QProgressBar, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(931, 674)
        MainWindow.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        icon = QIcon()
        icon.addFile(u"../../.designer/backup/icons/Republic_of_China_Police_Logo.svg.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet(u"background-color: rgb(10, 15, 27);\n"
"font-family: Noto Sans SC;\n"
"color:white;")
        MainWindow.setDocumentMode(False)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_9 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_8 = QVBoxLayout()
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_7 = QVBoxLayout()
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.Cate = QFrame(self.centralwidget)
        self.Cate.setObjectName(u"Cate")
        self.Cate.setStyleSheet(u"background-color: rgb(32, 40, 57);\n"
"border-radius: 7px;\n"
"font-size:15pt;")
        self.horizontalLayout = QHBoxLayout(self.Cate)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.label_2 = QLabel(self.Cate)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setStyleSheet(u"font-size:14pt;")

        self.verticalLayout_4.addWidget(self.label_2)

        self.input_image = QLabel(self.Cate)
        self.input_image.setObjectName(u"input_image")
        self.input_image.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self.input_image.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.input_image.setStyleSheet(u"background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:14pt;")
        self.input_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_4.addWidget(self.input_image)

        self.verticalLayout_4.setStretch(1, 1)

        self.horizontalLayout.addLayout(self.verticalLayout_4)

        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.label_5 = QLabel(self.Cate)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setStyleSheet(u"font-size:14pt;")

        self.verticalLayout_5.addWidget(self.label_5)

        self.input_image_2 = QLabel(self.Cate)
        self.input_image_2.setObjectName(u"input_image_2")
        self.input_image_2.setMinimumSize(QSize(0, 0))
        self.input_image_2.setMouseTracking(False)
        self.input_image_2.setStyleSheet(u"background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:14pt;")
        self.input_image_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_image_2.setWordWrap(False)

        self.verticalLayout_5.addWidget(self.input_image_2)

        self.verticalLayout_5.setStretch(1, 1)

        self.horizontalLayout.addLayout(self.verticalLayout_5)


        self.verticalLayout_7.addWidget(self.Cate)

        self.Cato2 = QFrame(self.centralwidget)
        self.Cato2.setObjectName(u"Cato2")
        self.Cato2.setStyleSheet(u"background-color: rgb(32, 40, 57);\n"
"border-radius: 7px;")
        self.horizontalLayout_4 = QHBoxLayout(self.Cato2)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.label_3 = QLabel(self.Cato2)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setStyleSheet(u"font-size:14pt;")

        self.verticalLayout_6.addWidget(self.label_3)

        self.original_image = QLabel(self.Cato2)
        self.original_image.setObjectName(u"original_image")
        self.original_image.setStyleSheet(u"background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:14pt;")
        self.original_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_6.addWidget(self.original_image)

        self.verticalLayout_6.setStretch(1, 1)

        self.horizontalLayout_4.addLayout(self.verticalLayout_6)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.label_4 = QLabel(self.Cato2)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setStyleSheet(u"font-size:14pt;")

        self.verticalLayout.addWidget(self.label_4)

        self.original_image_2 = QLabel(self.Cato2)
        self.original_image_2.setObjectName(u"original_image_2")
        self.original_image_2.setStyleSheet(u"background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:14pt;")
        self.original_image_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout.addWidget(self.original_image_2)

        self.verticalLayout.setStretch(1, 1)

        self.horizontalLayout_4.addLayout(self.verticalLayout)


        self.verticalLayout_7.addWidget(self.Cato2)


        self.verticalLayout_8.addLayout(self.verticalLayout_7)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.Start_button = QPushButton(self.centralwidget)
        self.Start_button.setObjectName(u"Start_button")
        self.Start_button.setStyleSheet(u"background-color: rgb(239, 98, 15);\n"
"font-size:12pt;\n"
"color: white;")
        icon1 = QIcon()
        icon1.addFile(u"icons/smart_toy_42dp_E8EAED_FILL0_wght400_GRAD0_opsz40.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.Start_button.setIcon(icon1)

        self.horizontalLayout_3.addWidget(self.Start_button)

        self.Start_button_2 = QPushButton(self.centralwidget)
        self.Start_button_2.setObjectName(u"Start_button_2")
        self.Start_button_2.setStyleSheet(u"background-color: rgb(239, 98, 15);\n"
"font-size:12pt;\n"
"color: white;")
        icon2 = QIcon()
        icon2.addFile(u"icons/delete_42dp_E8EAED_FILL0_wght400_GRAD0_opsz40.svg", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.Start_button_2.setIcon(icon2)

        self.horizontalLayout_3.addWidget(self.Start_button_2)


        self.verticalLayout_3.addLayout(self.horizontalLayout_3)

        self.progressBar = QProgressBar(self.centralwidget)
        self.progressBar.setObjectName(u"progressBar")
        self.progressBar.setStyleSheet(u"color: white;")
        self.progressBar.setValue(0)

        self.verticalLayout_3.addWidget(self.progressBar)


        self.verticalLayout_8.addLayout(self.verticalLayout_3)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.analyz_3 = QLabel(self.centralwidget)
        self.analyz_3.setObjectName(u"analyz_3")
        self.analyz_3.setMinimumSize(QSize(0, 0))
        self.analyz_3.setStyleSheet(u"background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:14pt;")
        self.analyz_3.setFrameShape(QFrame.Shape.NoFrame)
        self.analyz_3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.analyz_3.setMargin(10)

        self.horizontalLayout_6.addWidget(self.analyz_3)

        self.top1 = QLabel(self.centralwidget)
        self.top1.setObjectName(u"top1")
        self.top1.setMinimumSize(QSize(300, 0))
        self.top1.setToolTipDuration(-1)
        self.top1.setStyleSheet(u"background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:12pt;")
        self.top1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top1.setMargin(10)

        self.horizontalLayout_6.addWidget(self.top1)


        self.horizontalLayout_7.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.analyz_4 = QLabel(self.centralwidget)
        self.analyz_4.setObjectName(u"analyz_4")
        self.analyz_4.setMinimumSize(QSize(0, 0))
        self.analyz_4.setStyleSheet(u"background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:14pt;")
        self.analyz_4.setFrameShape(QFrame.Shape.NoFrame)
        self.analyz_4.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.analyz_4.setMargin(10)

        self.horizontalLayout_2.addWidget(self.analyz_4)

        self.top1_black_white = QLabel(self.centralwidget)
        self.top1_black_white.setObjectName(u"top1_black_white")
        self.top1_black_white.setMinimumSize(QSize(300, 0))
        self.top1_black_white.setStyleSheet(u"background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:12pt;")
        self.top1_black_white.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top1_black_white.setMargin(10)

        self.horizontalLayout_2.addWidget(self.top1_black_white)


        self.horizontalLayout_7.addLayout(self.horizontalLayout_2)


        self.verticalLayout_8.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.analyz_5 = QLabel(self.centralwidget)
        self.analyz_5.setObjectName(u"analyz_5")
        self.analyz_5.setStyleSheet(u"background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:14pt;")
        self.analyz_5.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.analyz_5.setMargin(10)

        self.horizontalLayout_9.addWidget(self.analyz_5)

        self.top2 = QLabel(self.centralwidget)
        self.top2.setObjectName(u"top2")
        self.top2.setMinimumSize(QSize(300, 0))
        self.top2.setStyleSheet(u"background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:12pt;")
        self.top2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top2.setMargin(10)

        self.horizontalLayout_9.addWidget(self.top2)


        self.horizontalLayout_12.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.analyz_7 = QLabel(self.centralwidget)
        self.analyz_7.setObjectName(u"analyz_7")
        self.analyz_7.setStyleSheet(u"background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:14pt;")
        self.analyz_7.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.analyz_7.setMargin(10)

        self.horizontalLayout_8.addWidget(self.analyz_7)

        self.top2_black_white = QLabel(self.centralwidget)
        self.top2_black_white.setObjectName(u"top2_black_white")
        self.top2_black_white.setMinimumSize(QSize(300, 0))
        self.top2_black_white.setStyleSheet(u"background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:12pt;")
        self.top2_black_white.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top2_black_white.setMargin(10)

        self.horizontalLayout_8.addWidget(self.top2_black_white)


        self.horizontalLayout_12.addLayout(self.horizontalLayout_8)


        self.verticalLayout_8.addLayout(self.horizontalLayout_12)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.analyz_6 = QLabel(self.centralwidget)
        self.analyz_6.setObjectName(u"analyz_6")
        self.analyz_6.setStyleSheet(u"background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:14pt;")
        self.analyz_6.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.analyz_6.setMargin(10)

        self.horizontalLayout_11.addWidget(self.analyz_6)

        self.top3 = QLabel(self.centralwidget)
        self.top3.setObjectName(u"top3")
        self.top3.setMinimumSize(QSize(300, 0))
        self.top3.setStyleSheet(u"background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:12pt;")
        self.top3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top3.setMargin(10)

        self.horizontalLayout_11.addWidget(self.top3)


        self.horizontalLayout_13.addLayout(self.horizontalLayout_11)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.analyz_8 = QLabel(self.centralwidget)
        self.analyz_8.setObjectName(u"analyz_8")
        self.analyz_8.setStyleSheet(u"background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:14pt;")
        self.analyz_8.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.analyz_8.setMargin(10)

        self.horizontalLayout_10.addWidget(self.analyz_8)

        self.top3_black_white = QLabel(self.centralwidget)
        self.top3_black_white.setObjectName(u"top3_black_white")
        self.top3_black_white.setMinimumSize(QSize(300, 0))
        self.top3_black_white.setStyleSheet(u"background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:12pt;")
        self.top3_black_white.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top3_black_white.setMargin(10)

        self.horizontalLayout_10.addWidget(self.top3_black_white)


        self.horizontalLayout_13.addLayout(self.horizontalLayout_10)


        self.verticalLayout_8.addLayout(self.horizontalLayout_13)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.analyz = QLabel(self.centralwidget)
        self.analyz.setObjectName(u"analyz")
        self.analyz.setStyleSheet(u"background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:14pt;")
        self.analyz.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_2.addWidget(self.analyz)


        self.horizontalLayout_5.addLayout(self.verticalLayout_2)

        self.analyz_2 = QLabel(self.centralwidget)
        self.analyz_2.setObjectName(u"analyz_2")
        font = QFont()
        font.setFamilies([u"Noto Sans SC"])
        font.setPointSize(14)
        self.analyz_2.setFont(font)
        self.analyz_2.setStyleSheet(u"background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:14pt;")
        self.analyz_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.horizontalLayout_5.addWidget(self.analyz_2)


        self.verticalLayout_8.addLayout(self.horizontalLayout_5)


        self.verticalLayout_9.addLayout(self.verticalLayout_8)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"\u95dc\u9632\u5716\u7ae0AI\u8fa8\u8b58", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"\u6b32\u8fa8\u8b58\u5370\u7ae0", None))
        self.input_image.setText(QCoreApplication.translate("MainWindow", u"\u6b32\u8fa8\u8b58\u5370\u7ae0", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"\u6b32\u8fa8\u8b58\u5370\u7ae0\u7279\u5fb5", None))
        self.input_image_2.setText(QCoreApplication.translate("MainWindow", u"\u6b32\u8fa8\u8b58\u5370\u7ae0\u7279\u5fb5", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"\u6700\u76f8\u4f3c\u5370\u7ae0", None))
        self.original_image.setText(QCoreApplication.translate("MainWindow", u"\u6700\u76f8\u4f3c\u5370\u7ae0", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"\u6700\u76f8\u4f3c\u5370\u7ae0\u7279\u5fb5", None))
        self.original_image_2.setText(QCoreApplication.translate("MainWindow", u"\u6700\u76f8\u4f3c\u5370\u7ae0\u7279\u5fb5", None))
        self.Start_button.setText(QCoreApplication.translate("MainWindow", u"\u958b\u59cb", None))
        self.Start_button_2.setText(QCoreApplication.translate("MainWindow", u"\u6e05\u9664 ", None))
        self.analyz_3.setText(QCoreApplication.translate("MainWindow", u"Top 1", None))
        self.top1.setText("")
        self.analyz_4.setText(QCoreApplication.translate("MainWindow", u"Top 1", None))
        self.top1_black_white.setText("")
        self.analyz_5.setText(QCoreApplication.translate("MainWindow", u"Top 2", None))
        self.top2.setText("")
        self.analyz_7.setText(QCoreApplication.translate("MainWindow", u"Top 2", None))
        self.top2_black_white.setText("")
        self.analyz_6.setText(QCoreApplication.translate("MainWindow", u"Top 3", None))
        self.top3.setText("")
        self.analyz_8.setText(QCoreApplication.translate("MainWindow", u"Top 3", None))
        self.top3_black_white.setText("")
        self.analyz.setText(QCoreApplication.translate("MainWindow", u"\u985e\u5225\u540d\u7a31", None))
        self.analyz_2.setText(QCoreApplication.translate("MainWindow", u"\u5206\u6790\u7d50\u8ad6\n"
"\u5b57\u9ad4\u5dee\u7570: \u540c\u5b57\u9ad4 (\u4e0d\u540c\u5b57\u9ad4)\n"
"\u5b57\u6a23\u5dee\u7570: \u540c\u5b57\u6a23 (\u4e0d\u540c\u5b57\u6a23_\n"
"\u5b57\u8ddd\u5dee\u7570: \u5b57\u8ddd\u76f8\u540c(\u5b57\u8ddd\u4e0d\u540c)\n"
"\u884c\u8ddd\u5dee\u7570\u6027: \u884c\u8ddd\u76f8\u540c(\u884c\u8ddd\u4e0d\u540c)\n"
"", None))
    # retranslateUi

