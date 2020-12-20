import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QFileDialog
from PIL.ImageQt import ImageQt
import test_1
import os
import torch
from torch.utils.data import DataLoader
import math
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage, Compose, Normalize
import torch.utils.data.dataset as dataset
from PIL import Image
import tool
from get_model_new import Model
import random
import tdwy_9 as tdwy
import tdwy_deblur
import test_deblur_1
import test_deblur_4
import test_deblurring_single
import test_deblurring_serious
import test_tdwy_full_slight
import test_tdwy_full_serious
import test_tdwy_full_slight_single
import test_tdwy_full_serious_single

global filename_2, path_a, state_n, pnsr, ssim, state_m, filename_o, state_d
state_n = '单张'
state_m = '明亮'
state_d = '整张'
unloader = transforms.ToPILImage()
# a1, b1, c1, model_test=model(model_path,'model_best.pth',retrain=False).load_model()
# model = model.module
# model_test.eval()
eval_loss = 0
pnsr = 0
count = 0
# print(len(train_loader))
model_name = 'model-9.pth'
model = tdwy.TDWY_NET()
model_path = './model/'
result_before = "./dped_patch/net_9/Original/"
result_after = "./dped_patch/net_9/BoostLight/"
result_label = './dped_patch/net_9/gt_patch/'
result_label = '../test_data/DPED/canon/'
result_after_light1 = "./dped_patch/net_out/"
tool.Mkdir(result_after_light1)
filename_o = result_after_light1
tool.Mkdir(result_before)
tool.Mkdir(result_after)
tool.Mkdir(result_label)


class Ui_Dialog(QWidget):  # (object)
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(760, 590)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(570, 70, 61, 31))
        self.pushButton.setObjectName("pushButton")
        self.textEdit = QtWidgets.QTextEdit(Dialog)
        self.textEdit.setGeometry(QtCore.QRect(220, 70, 331, 41))
        self.textEdit.setObjectName("textEdit")
        self.textEdit_2 = QtWidgets.QTextEdit(Dialog)
        self.textEdit_2.setGeometry(QtCore.QRect(220, 120, 331, 41))
        self.textEdit_2.setObjectName("textEdit_2")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(570, 120, 61, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(360, 420, 75, 41))
        self.pushButton_3.setObjectName("pushButton_3")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(50, 230, 211, 171))
        self.label.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(290, 230, 201, 171))
        self.label_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(140, 80, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_3.setFont(font)
        self.label_3.setTextFormat(QtCore.Qt.AutoText)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(140, 130, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_4.setFont(font)
        self.label_4.setTextFormat(QtCore.Qt.AutoText)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(300, 480, 71, 21))
        self.label_5.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.comboBox = QtWidgets.QComboBox(Dialog)
        self.comboBox.setGeometry(QtCore.QRect(50, 30, 61, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.label_6 = QtWidgets.QLabel(Dialog)
        self.label_6.setGeometry(QtCore.QRect(450, 480, 71, 21))
        self.label_6.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(Dialog)
        self.label_7.setGeometry(QtCore.QRect(240, 480, 54, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(Dialog)
        self.label_8.setGeometry(QtCore.QRect(390, 480, 54, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.comboBox_2 = QtWidgets.QComboBox(Dialog)
        self.comboBox_2.setGeometry(QtCore.QRect(120, 30, 71, 22))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.label_9 = QtWidgets.QLabel(Dialog)
        self.label_9.setGeometry(QtCore.QRect(520, 230, 201, 171))
        self.label_9.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_9.setText("")
        self.label_9.setObjectName("label_9")
        self.textEdit_3 = QtWidgets.QTextEdit(Dialog)
        self.textEdit_3.setGeometry(QtCore.QRect(220, 170, 331, 41))
        self.textEdit_3.setObjectName("textEdit_3")
        self.pushButton_4 = QtWidgets.QPushButton(Dialog)
        self.pushButton_4.setGeometry(QtCore.QRect(570, 170, 61, 31))
        self.pushButton_4.setObjectName("pushButton_4")
        self.label_10 = QtWidgets.QLabel(Dialog)
        self.label_10.setGeometry(QtCore.QRect(140, 180, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_10.setFont(font)
        self.label_10.setTextFormat(QtCore.Qt.AutoText)
        self.label_10.setObjectName("label_10")
        self.comboBox_3 = QtWidgets.QComboBox(Dialog)
        self.comboBox_3.setGeometry(QtCore.QRect(200, 30, 61, 22))
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")

        self.comboBox.currentIndexChanged.connect(self.select_change)
        self.comboBox_2.currentIndexChanged.connect(self.select_change)
        self.comboBox_3.currentIndexChanged.connect(self.select_change)
        self.pushButton.clicked.connect(self.open_run)
        self.pushButton_2.clicked.connect(self.open_run)
        self.pushButton_3.clicked.connect(self.open_run)
        self.pushButton_4.clicked.connect(self.open_run)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def open_run(self):
        global filename_2, path_a, pnsr, ssim, filename_o
        sender = self.sender()
        if state_n == '多张':
            if sender == self.pushButton:
                path_a = QFileDialog.getExistingDirectory(None, '打开文件')
                if path_a:
                    self.textEdit.setText(path_a)
            elif sender == self.pushButton_2:
                filename_2 = QFileDialog.getExistingDirectory(None, '打开文件')
                if filename_2:
                    self.textEdit_2.setText(filename_2)
            elif sender == self.pushButton_4:
                filename_o = QFileDialog.getExistingDirectory(None, '打开文件')
                if filename_o:
                    filename_o = filename_o+"/"
                    self.textEdit_3.setText(filename_o)
            elif sender == self.pushButton_3:
                if state_m == '明亮':
                    test_light_n()
                    self.label_5.setText('%f' % float(pnsr))
                    self.label_6.setText('%f' % float(ssim))
                elif state_m == '中度模糊':
                    if state_d == '整张':
                        test_deblur_1.test(path_a, filename_2, filename_o)
                        self.label_5.setText('%f' % float(test_deblur_1.pnsr))
                        self.label_6.setText('%f' % float(test_deblur_1.ssim))
                    else:
                        test_tdwy_full_slight.test(path_a, filename_o)
                elif state_m == '严重模糊':
                    if state_d == '整张':
                        test_deblur_4.test(path_a, filename_2, filename_o)
                        self.label_5.setText('%f' % float(test_deblur_4.pnsr))
                        self.label_6.setText('%f' % float(test_deblur_4.ssim))
                    else:
                        test_tdwy_full_serious.test(path_a, filename_o)
        elif state_n == '单张':
            if sender == self.pushButton:
                path_a, _filter = QFileDialog.getOpenFileName(None, '打开文件')
                if path_a:
                    self.textEdit.setText(path_a)
                    # img_1 = Image.open(path_a)
                    self.label.setPixmap(
                        QtGui.QPixmap(path_a).scaled(
                            200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            elif sender == self.pushButton_2:
                filename_2, _filter = QFileDialog.getOpenFileName(None, '打开文件')
                if filename_2:
                    self.textEdit_2.setText(filename_2)
                    # img_2 = Image.open(filename_2)
                    self.label_2.setPixmap(
                        QtGui.QPixmap(filename_2).scaled(
                            200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            elif sender == self.pushButton_4:
                filename_o = QFileDialog.getExistingDirectory(None, '打开文件')
                if filename_o:
                    filename_o = filename_o + "/"
                    self.textEdit_3.setText(filename_o)
            elif sender == self.pushButton_3:
                if state_m == '明亮':
                    img_2 = test_light_1()
                    img_2o = ImageQt(img_2)
                    self.label_9.setPixmap(
                        QtGui.QPixmap.fromImage(img_2o).scaled(
                            200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    self.label_5.setText('%f' % float(pnsr))
                    self.label_6.setText('%f' % float(ssim))
                elif state_m == '中度模糊':
                    if state_d == '整张':
                        img_2 = test_deblurring_single.test(path_a, filename_2, filename_o)
                        img_2o = ImageQt(img_2)
                        self.label_9.setPixmap(
                            QtGui.QPixmap.fromImage(img_2o).scaled(
                                200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                        self.label_5.setText('%f' % float(test_deblurring_single.pnsr))
                        self.label_6.setText('%f' % float(test_deblurring_single.ssim))
                    else:
                        img_2 = test_tdwy_full_slight_single.test(path_a, filename_o)
                        img_2o = ImageQt(img_2)
                        self.label_9.setPixmap(
                            QtGui.QPixmap.fromImage(img_2o).scaled(
                                200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                elif state_m == '严重模糊':
                    if state_d == '整张':
                        img_2 = test_deblurring_serious.test(path_a, filename_2, filename_o)
                        img_2o = ImageQt(img_2)
                        self.label_9.setPixmap(
                            QtGui.QPixmap.fromImage(img_2o).scaled(
                                200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                        self.label_5.setText('%f' % float(test_deblurring_serious.pnsr))
                        self.label_6.setText('%f' % float(test_deblurring_serious.ssim))
                    else:
                        img_2 = test_tdwy_full_serious_single.test(path_a, filename_o)
                        img_2o = ImageQt(img_2)
                        self.label_9.setPixmap(
                            QtGui.QPixmap.fromImage(img_2o).scaled(
                                200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def select_change(self):
        global state_n, state_m, state_d
        state_n = self.comboBox.currentText()
        state_m = self.comboBox_2.currentText()
        state_d = self.comboBox_3.currentText()
        if state_n == '单张':
            self.label.setStyleSheet("background-color: rgb(255, 255, 255);")
            self.label_2.setStyleSheet("background-color: rgb(255, 255, 255);")
            self.label_9.setStyleSheet("background-color: rgb(255, 255, 255);")
        else:
            self.label.setStyleSheet("background-color: rgb(240, 240, 240);")
            self.label_2.setStyleSheet("background-color: rgb(240, 240, 240);")
            self.label_9.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.label.clear()
        self.label_2.clear()
        self.label_9.clear()
        self.textEdit.clear()
        self.textEdit_2.clear()
        self.textEdit_3.clear()
        self.label_5.clear()
        self.label_6.clear()

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "打开"))
        self.pushButton_2.setText(_translate("Dialog", "打开"))
        self.pushButton_3.setText(_translate("Dialog", "运行"))
        self.label_3.setText(_translate("Dialog", "图片路径"))
        self.label_4.setText(_translate("Dialog", "匹配路径"))
        self.comboBox.setItemText(0, _translate("Dialog", "单张"))
        self.comboBox.setItemText(1, _translate("Dialog", "多张"))
        self.label_7.setText(_translate("Dialog", " pnsr："))
        self.label_8.setText(_translate("Dialog", " ssim："))
        self.comboBox_2.setItemText(0, _translate("Dialog", "明亮"))
        self.comboBox_2.setItemText(1, _translate("Dialog", "中度模糊"))
        self.comboBox_2.setItemText(2, _translate("Dialog", "严重模糊"))
        self.pushButton_4.setText(_translate("Dialog", "打开"))
        self.label_10.setText(_translate("Dialog", "保存路径"))
        self.comboBox_3.setItemText(0, _translate("Dialog", "整张"))
        self.comboBox_3.setItemText(1, _translate("Dialog", "分割"))


def psnr1(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    # print(image)
    # image = transforms.Normalize([-1, -1, -1], [2, 2, 2])(image)
    # print(image)
    image = unloader(image)
    # image = transforms.Normalize([-1, -1, -1], [-1, -1, -1])(image)
    return image


def test_light_1():
    global filename_2, path_a, pnsr, ssim, filename_o
    a1, b1, c1, model_test = Model(
        model_path, model_name, retrain=False, load_parameters=True).load_model(model)
    print(a1)
    model_test.cuda()
    model_test.eval()
    with torch.no_grad():
        img = Image.open(path_a)
        img = ToTensor()(img)
        img = img.unsqueeze(0)
        label = Image.open(filename_2)
        label = ToTensor()(label)
        label = label.unsqueeze(0)
        if torch.cuda.is_available():
            img = img.cuda()
        output = model_test(img)
        output = torch.clamp(output, 0, 1)
        out_img = tensor_to_PIL(output)
        i = np.array(out_img)
        out_img.save(filename_o + "{}.png".format(1))
        label = tensor_to_PIL(label)
        ii = np.array(label)
        pnsr = psnr1(i, ii)
        ssim = tool.calculate_ssim(i, ii)
        print('psnr`::%f' % float(pnsr), 'ssim`::%f' % float(ssim))
    return out_img


def test_light_n():
    global path_a, pnsr, ssim
    test_data = test_1.My_dataset(img_path=path_a)
    train_loader = DataLoader(test_data, batch_size=1)
    a1, b1, c1, model_test = Model(
        model_path, model_name, retrain=False, load_parameters=True).load_model(model)
    print(a1)
    model_test.cuda()
    model_test.eval()
    pnsr = 0
    ssim = 0
    count = 0
    zero = 0
    with torch.no_grad():
        for img, label, img_n in train_loader:
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
            output = model_test(img)
            img_save = tensor_to_PIL(img)
            img_save.save(result_before + '{}.png'.format(img_n))
            # print(output)
            output = torch.clamp(output, 0, 1)
            out_img = tensor_to_PIL(output)
            i = np.array(out_img)
            # print(i)
            img_n = list(img_n)[0]
            out_img.save(result_after + "{}.png".format(img_n))
            label = tensor_to_PIL(label)
            label.save(result_label + '{}.png'.format(img_n))
            ii = np.array(label)
            count += 1
            print(count)
            a = psnr1(i, ii)
            if a < 48:
                zero += 1
                ssim += tool.calculate_ssim(i, ii)
                pnsr += psnr1(i, ii)
    pnsr = float(pnsr / zero)
    ssim = float(ssim / zero)
    print('mean_psnr`::%f' % float(pnsr), 'mean_ssim`::%f' % float(ssim))


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    widget = QtWidgets.QWidget()
    ui = Ui_Dialog()
    ui.setupUi(widget)
    widget.show()
    sys.exit(app.exec_())
