# -*- encoding: utf-8 -*-
import os
import re
import sys
import matplotlib.pyplot as plt
import pandas as pd


def read_file(filename, start_time, end_time):
   if filename.endswith(".xlsx") or filename.endswith(".xls"):
      df = pd.read_excel(filename, sheet_name=0, skiprows=0, na_values=[])
   elif filename.endswith(".csv"):
      df = pd.read_csv(filename)
   
   res = []
   for index, row in df.items():
      for i in range(len(row)):
         dtime = row[i] / 10**9
         if start_time <= dtime <= end_time:
            res.append(dtime - start_time)

   return res

def visualize(all_data, start_time, end_time, scale):
   for k, data in enumerate(all_data):
      x, y = [], []
      for dtime in data:
         x.append((dtime, dtime))
         y.append((k+0.9, k+0.1))

      for i in range(len(x)):
         plt.plot(x[i], y[i], color='k')

   plt.xlim(0, end_time - start_time)
   plt.ylim(0, len(all_data))
   plt.savefig(fname="result.svg",format="svg")
   plt.show()


if __name__ == '__main__':
   print("当前路径为: %s" % os.getcwd())
   all_data = []
   length = []
   while 1:
      print("将你需要处理的excel或csv文件放到本文件相同的路径下, 并输入文件名(或者直接输入绝对路径):")
      filename = input()
      while not os.path.exists(filename):
         print("文件不存在，请重新输入:")
         filename = input()
      else:
         print("%s 文件读取成功！正在处理..." % filename)

      while 1:
         try:
            print("输入开始时间(确认输入的时间是一个合理的数字):")
            start_time = int(input())
            print("开始时间: %ds" % start_time)
            print("输入结束时间(确认输入的时间是一个合理的数字):")
            end_time = int(input())
            print("结束时间: %ds" % end_time)
            break
         except:
            print("输入有误请重新输入!")
      
      data = read_file(filename, start_time, end_time)
      all_data.append(data)
      length.append([start_time, end_time, len(data)])
      
      print("输入任意字母继续输入下一个文件, 直接按下回车结束输入文件:")

      next = input()
      if next.isalpha():
         continue
      # print("输入刻度(确认输入的刻度是一个合理的数字):")
      # scale = int(input())
      # print("刻度: %ds" % scale)
      break

   print("正在生成图片...")
   for i,k in enumerate(length):
      print(f'The {i+1} segment from {k[0]}s to {k[1]}s contains {k[2]} licks')
   
   visualize(all_data, start_time, end_time, 1)