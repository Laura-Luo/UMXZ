import os

# project path
PRO_PATH = os.path.abspath(os.path.dirname(__file__))

# 乐曲片段文件夹
TRAIN_DIR_1 = os.path.join(PRO_PATH, r'data\train') #
TEST_DIR_1 = os.path.join(PRO_PATH, r'data\test') #
PRED_DIR = os.path.join(PRO_PATH, r'data\predict') #
XML_DIR = os.path.join(PRO_PATH, r'转谱\MusicXml')
New_DIR = os.path.join(PRO_PATH, r'转谱/ZongAudio')
CUT_DIR = os.path.join(PRO_PATH, r'转谱/Audio')
TRAIN_DIR_2 = os.path.join(PRO_PATH, r'data2/train')
VALID_DIR_2 = os.path.join(PRO_PATH, r'data2/valid')
TEST_DIR_2 = os.path.join(PRO_PATH, r'data2/test')


