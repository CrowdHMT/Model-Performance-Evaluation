# @FunctionName: Check User Model
# @Author: Wangyuzhan
# @Time: 2023/12/7
from checkmodel_util import test
from checkmodel_util import model_user

import os

def ChangeUserModelCode(file,old_str,new_str):
    """
    将替换的字符串写到一个新的文件中，然后将原文件删除，新文件改为原来文件的名字
    :param file: 文件路径
    :param old_str: 需要替换的字符串
    :param new_str: 替换的字符串
    :return: None
    """
    with open(file, "r", encoding="utf-8") as f1,open("%s.bak" % file, "w", encoding="utf-8") as f2:
        for line in f1:
            if old_str in line:
                line = line.replace(old_str, new_str)
            f2.write(line)
    os.remove(file)
    os.rename("%s.bak" % file, file)
 
if __name__ == "__main__":
    is_error = 0
    ChangeUserModelCode("checkmodel_util.py", "AlexNet", "UserModel")
    try:
        test_result = test()
        model, input = model_user()
        x = model(input)
        print(x.size())
    except:
        is_error = 1
        print("model test fail")
    if is_error == 0 and test_result == x.size():
        print("model test pass")
    

