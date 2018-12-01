import pymysql
import xlwt


class DataBase():
    def __init__(self, host, username, password, db_name):
        self.db = pymysql.connect(host, username, password, db_name)
        self.cursor = self.db.cursor()

    def select(self, sql_commend):
        try:
            self.cursor.execute(sql_commend)
            results = self.cursor.fetchall()
            return results
        except:
            print("Error: unable to fetch data")

# 连接到数据库
comment_db = DataBase('', '', '', '')

jingwutuan_ios = comment_db.select('select comment,rate from appstore')
jingwutuan_taptap = comment_db.select('select comment,rate from taptap')

comment_workbook = xlwt.Workbook()
positive_comment_sheet = comment_workbook.add_sheet("positive_comment")
negative_comment_sheet = comment_workbook.add_sheet("negative_comment")
negative_row_index = 0
positive_row_index = 0
for row in appstore:
    if row[1] < 0.7:
        negative_comment_sheet.write(negative_row_index, 0, row[0])
        negative_row_index = negative_row_index + 1
    else:
        positive_comment_sheet.write(positive_row_index, 0, row[0])
        positive_row_index = positive_row_index + 1
for row in taptap:
    if row[1] < 0.7:
        negative_comment_sheet.write(negative_row_index, 0, row[0])
        negative_row_index = negative_row_index + 1
    else:
        positive_comment_sheet.write(positive_row_index, 0, row[0])
        positive_row_index = positive_row_index + 1
comment_workbook.save("comment.xls")
