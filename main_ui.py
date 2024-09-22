# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\ui_main.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from utils import resource_path_1  # Импортируйте функцию
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(905, 619)
        MainWindow.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        icon = QtGui.QIcon()
        icon_1 = resource_path_1("\\icons/Republic_of_China_Police_Logo.svg.png")
        icon.addPixmap(QtGui.QPixmap(icon_1), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("background-color: rgb(10, 15, 27);\n"
"font-family: Noto Sans SC;\n"
"color:white;")
        MainWindow.setDocumentMode(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.Cate = QtWidgets.QFrame(self.centralwidget)
        self.Cate.setStyleSheet("background-color: rgb(32, 40, 57);\n"
"border-radius: 7px;\n"
"font-size:15pt;")
        self.Cate.setObjectName("Cate")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.Cate)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_2 = QtWidgets.QLabel(self.Cate)
        self.label_2.setStyleSheet("font-size:12pt;")
        self.label_2.setObjectName("label_2")
        self.verticalLayout_4.addWidget(self.label_2)
        self.input_image = QtWidgets.QLabel(self.Cate)
        self.input_image.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.input_image.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.input_image.setStyleSheet("background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:12pt;")
        self.input_image.setAlignment(QtCore.Qt.AlignCenter)
        self.input_image.setObjectName("input_image")
        self.verticalLayout_4.addWidget(self.input_image)
        self.verticalLayout_4.setStretch(1, 1)
        self.horizontalLayout.addLayout(self.verticalLayout_4)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_5 = QtWidgets.QLabel(self.Cate)
        self.label_5.setStyleSheet("font-size:12pt;")
        self.label_5.setObjectName("label_5")
        self.verticalLayout_5.addWidget(self.label_5)
        self.image2 = QtWidgets.QLabel(self.Cate)
        self.image2.setMouseTracking(False)
        self.image2.setStyleSheet("background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:12pt;")
        self.image2.setAlignment(QtCore.Qt.AlignCenter)
        self.image2.setWordWrap(False)
        self.image2.setObjectName("image2")
        self.verticalLayout_5.addWidget(self.image2)
        self.verticalLayout_5.setStretch(1, 1)
        self.horizontalLayout.addLayout(self.verticalLayout_5)
        self.verticalLayout_7.addWidget(self.Cate)
        self.Cato2 = QtWidgets.QFrame(self.centralwidget)
        self.Cato2.setStyleSheet("background-color: rgb(32, 40, 57);\n"
"border-radius: 7px;")
        self.Cato2.setObjectName("Cato2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.Cato2)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_3 = QtWidgets.QLabel(self.Cato2)
        self.label_3.setStyleSheet("font-size:12pt;")
        self.label_3.setObjectName("label_3")
        self.verticalLayout_6.addWidget(self.label_3)
        self.model_image = QtWidgets.QLabel(self.Cato2)
        self.model_image.setStyleSheet("background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:12pt;")
        self.model_image.setAlignment(QtCore.Qt.AlignCenter)
        self.model_image.setObjectName("model_image")
        self.verticalLayout_6.addWidget(self.model_image)
        self.verticalLayout_6.setStretch(1, 1)
        self.horizontalLayout_4.addLayout(self.verticalLayout_6)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_4 = QtWidgets.QLabel(self.Cato2)
        self.label_4.setStyleSheet("font-size:12pt;")
        self.label_4.setObjectName("label_4")
        self.verticalLayout.addWidget(self.label_4)
        self.label = QtWidgets.QLabel(self.Cato2)
        self.label.setStyleSheet("background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:12pt;")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.verticalLayout.setStretch(1, 1)
        self.horizontalLayout_4.addLayout(self.verticalLayout)
        self.verticalLayout_7.addWidget(self.Cato2)
        self.verticalLayout_9.addLayout(self.verticalLayout_7)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.Model_choose_button = QtWidgets.QPushButton(self.centralwidget)
        self.Model_choose_button.setStyleSheet("background-color: rgb(239, 98, 15);\n"
"font-size:10pt;\n"
"color: white;")
        icon1 = QtGui.QIcon()
        icon_2 = resource_path_1(".\\icons/smart_toy_42dp_E8EAED_FILL0_wght400_GRAD0_opsz40.svg")
        icon1.addPixmap(QtGui.QPixmap(icon_2), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Model_choose_button.setIcon(icon1)
        self.Model_choose_button.setObjectName("Model_choose_button")
        self.horizontalLayout_2.addWidget(self.Model_choose_button)
        self.Crop_button = QtWidgets.QPushButton(self.centralwidget)
        self.Crop_button.setStyleSheet("background-color: rgb(239, 98, 15);\n"
"font-size:10pt;\n"
"color: white;")
        icon2 = QtGui.QIcon()
        icon_3 = resource_path_1(".\\icons/crop_rotate_42dp_E8EAED_FILL0_wght400_GRAD0_opsz40.svg")
        icon2.addPixmap(QtGui.QPixmap(icon_3), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Crop_button.setIcon(icon2)
        self.Crop_button.setObjectName("Crop_button")
        self.horizontalLayout_2.addWidget(self.Crop_button)
        self.Generator_button = QtWidgets.QPushButton(self.centralwidget)
        self.Generator_button.setStyleSheet("background-color: rgb(239, 98, 15);\n"
"font-size:10pt;\n"
"color: white;")
        icon3 = QtGui.QIcon()
        icon_4 = resource_path_1(".\\icons/token_42dp_E8EAED_FILL0_wght400_GRAD0_opsz40.svg")
        icon3.addPixmap(QtGui.QPixmap(icon_4), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Generator_button.setIcon(icon3)
        self.Generator_button.setObjectName("Generator_button")
        self.horizontalLayout_2.addWidget(self.Generator_button)
        self.Learn_button = QtWidgets.QPushButton(self.centralwidget)
        self.Learn_button.setStyleSheet("background-color: rgb(239, 98, 15);\n"
"font-size:10pt;\n"
"color: white;")
        icon4 = QtGui.QIcon()
        icon_5 = resource_path_1(".\\icons/account_tree_24dp_E8EAED_FILL0_wght400_GRAD0_opsz24.svg")
        icon4.addPixmap(QtGui.QPixmap(icon_5), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Learn_button.setIcon(icon4)
        self.Learn_button.setObjectName("Learn_button")
        self.horizontalLayout_2.addWidget(self.Learn_button)
        self.verticalLayout_8.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.Start_button = QtWidgets.QPushButton(self.centralwidget)
        self.Start_button.setStyleSheet("background-color: rgb(239, 98, 15);\n"
"font-size:10pt;\n"
"color: white;")
        icon5 = QtGui.QIcon()
        icon_6 = resource_path_1(".\\icons/task_alt_42dp_E8EAED_FILL0_wght400_GRAD0_opsz40.svg")
        icon5.addPixmap(QtGui.QPixmap(icon_6), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Start_button.setIcon(icon5)
        self.Start_button.setObjectName("Start_button")
        self.horizontalLayout_3.addWidget(self.Start_button)
        self.Test_button = QtWidgets.QPushButton(self.centralwidget)
        self.Test_button.setStyleSheet("background-color: rgb(239, 98, 15);\n"
"font-size:10pt;\n"
"color: white;")
        icon6 = QtGui.QIcon()
        icon_7 = resource_path_1(".\\icons/empty_dashboard_42dp_E8EAED_FILL0_wght400_GRAD0_opsz40.svg")
        icon6.addPixmap(QtGui.QPixmap(icon_7), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Test_button.setIcon(icon6)
        self.Test_button.setObjectName("Test_button")
        self.horizontalLayout_3.addWidget(self.Test_button)
        self.Save_button = QtWidgets.QPushButton(self.centralwidget)
        self.Save_button.setStyleSheet("background-color: rgb(239, 98, 15);\n"
"font-size:10pt;\n"
"color: white;")
        icon7 = QtGui.QIcon()
        icon_8 = resource_path_1(".\\icons/book_4_24dp_E8EAED_FILL0_wght400_GRAD0_opsz24.svg")
        icon7.addPixmap(QtGui.QPixmap(icon_8), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Save_button.setIcon(icon7)
        self.Save_button.setObjectName("Save_button")
        self.horizontalLayout_3.addWidget(self.Save_button)
        self.Clear_button = QtWidgets.QPushButton(self.centralwidget)
        self.Clear_button.setStyleSheet("background-color: rgb(239, 98, 15);\n"
"font-size:10pt;\n"
"color: white;")
        icon8 = QtGui.QIcon()
        icon_9 = resource_path_1(".\\icons/delete_42dp_E8EAED_FILL0_wght400_GRAD0_opsz40.svg")
        icon8.addPixmap(QtGui.QPixmap(icon_9), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Clear_button.setIcon(icon8)
        self.Clear_button.setObjectName("Clear_button")
        self.horizontalLayout_3.addWidget(self.Clear_button)
        self.verticalLayout_8.addLayout(self.horizontalLayout_3)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setStyleSheet("color: white;")
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_3.addWidget(self.progressBar)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.analyz = QtWidgets.QLabel(self.centralwidget)
        self.analyz.setStyleSheet("background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:15pt;")
        self.analyz.setAlignment(QtCore.Qt.AlignCenter)
        self.analyz.setObjectName("analyz")
        self.verticalLayout_2.addWidget(self.analyz)
        self.analyz_2 = QtWidgets.QLabel(self.centralwidget)
        self.analyz_2.setStyleSheet("background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:15pt;")
        self.analyz_2.setAlignment(QtCore.Qt.AlignCenter)
        self.analyz_2.setObjectName("analyz_2")
        self.verticalLayout_2.addWidget(self.analyz_2)
        self.horizontalLayout_5.addLayout(self.verticalLayout_2)
        self.analyz_3 = QtWidgets.QLabel(self.centralwidget)
        self.analyz_3.setStyleSheet("background-color: rgba(255, 255, 255, 30);\n"
"border:1px solid rgba(255, 255, 255, 40);\n"
"border-radius: 7px;\n"
"font-size:15pt;")
        self.analyz_3.setAlignment(QtCore.Qt.AlignCenter)
        self.analyz_3.setObjectName("analyz_3")
        self.horizontalLayout_5.addWidget(self.analyz_3)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.verticalLayout_3.setStretch(1, 1)
        self.verticalLayout_8.addLayout(self.verticalLayout_3)
        self.verticalLayout_9.addLayout(self.verticalLayout_8)
        self.verticalLayout_10.addLayout(self.verticalLayout_9)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "關防圖章AI辨識"))
        self.label_2.setText(_translate("MainWindow", "原始印章"))
        self.input_image.setText(_translate("MainWindow", "原始印章"))
        self.label_5.setText(_translate("MainWindow", "原始印章的特徵"))
        self.image2.setText(_translate("MainWindow", "原始印章的特徵"))
        self.label_3.setText(_translate("MainWindow", "與原始印章最相似的照片"))
        self.model_image.setText(_translate("MainWindow", "與原始印章最相似的照片"))
        self.label_4.setText(_translate("MainWindow", "原始印章的特徵"))
        self.label.setText(_translate("MainWindow", "原始印章的特徵"))
        self.Model_choose_button.setText(_translate("MainWindow", "選擇模型"))
        self.Crop_button.setText(_translate("MainWindow", "來源"))
        self.Generator_button.setText(_translate("MainWindow", "生成機"))
        self.Learn_button.setText(_translate("MainWindow", "模型的特性"))
        self.Start_button.setText(_translate("MainWindow", "開始"))
        self.Test_button.setText(_translate("MainWindow", "元數據"))
        self.Save_button.setText(_translate("MainWindow", "類別"))
        self.Clear_button.setText(_translate("MainWindow", "清除"))
        self.analyz.setText(_translate("MainWindow", "印章類別"))
        self.analyz_2.setText(_translate("MainWindow", "相似度百分比"))
        self.analyz_3.setText(_translate("MainWindow", "分析結論"))
