import pymysql
from tkinter import *
from tkinter import messagebox

##전역 변수부
#DB 관련
window, edtFrame,listFrame =None, None, None
conn, cur = None, None
data1,data2,data3,data4,data5 = "","","","",""
IP = '127.0.0.1'
USER = 'root'
PASSWORD = '1234'
DB = 'photo_db'
sql = ""

def insertData() :
    global conn,cur,data1,data2,data3,data4,data5,sql

    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB , charset='utf8')
    cur = conn.cursor()

    data1 = edt1.get()
    data2 = edt2.get()
    data3 = edt3.get()
    data4 = edt4.get()
    data5 = edt5.get()
    try:
        sql = "INSERT INTO member_table(m_id,m_pass,m_name,m_phone,m_email) VALUES('"
        sql += data1 + "','" + data2 + "','" + data3 + "','" + data4 + "','" + data5 +"')"
        cur.execute(sql)
    except :
        messagebox.showerror('오류', '데이터 입력 오류가 발생함')
    else:
        messagebox.showinfo('성공', '데이터 입력 성공')

    conn.commit()
    conn.close()


def deleteData() :
    global conn,cur,data1,sql

    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB , charset='utf8')
    cur = conn.cursor()

    data1 = edt1.get()

    try:
        sql = "DELETE FROM photo_table WHERE p_upUser = '" + data1 + "'"
        cur.execute(sql)
        sql = "DELETE FROM member_table WHERE m_id = '" + data1 + "'"
        cur.execute(sql)
    except :
        messagebox.showerror('오류', '데이터 입력 오류가 발생함')
    else:
        messagebox.showinfo('성공', '데이터 입력 성공')
    conn.commit()
    conn.close()

def editData() :
    global conn,cur,data1,data2,data3,data4,data5,sql

    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB , charset='utf8')
    cur = conn.cursor()

    data1 = edt1.get()
    data2 = edt2.get()
    data3 = edt3.get()
    data4 = edt4.get()
    data5 = edt5.get()

    try:
        sql = "UPDATE member_table SET m_pass = '" + data2 + "',m_name = '" + data3 + "',m_phone='" + data4
        sql += "',m_email='" + data5 + "'WHERE m_id ='" + data1 + "'"
        cur.execute(sql)
    except :
        messagebox.showerror('오류', '데이터 입력 오류가 발생함')
    else:
        messagebox.showinfo('성공', '데이터 입력 성공')
    conn.commit()
    conn.close()

def selectData():
    global conn, cur, data1, data2, data3, data4,data5,sql
    strData1, strData2, strData3, strData4, strData5 = [],[],[],[],[]
    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()
    cur.execute("SELECT * FROM member_table")
    strData1.append("                 ID")
    strData2.append("           Password")
    strData3.append("                이름")
    strData4.append("             전화번호")
    strData5.append("               이메일")
    strData1.append("=" * 20);strData2.append("=" * 20);strData3.append("=" * 20);
    strData4.append("=" * 20);strData5.append("=" * 20)
    while (True) :
        row = cur.fetchone()
        if row == None:
            break;
        strData1.append(row[0]); strData2.append(row[1]); strData3.append(row[2]);
        strData4.append(row[3]); strData5.append(row[4]);

    listData1.delete(0, listData1.size() - 1)
    listData2.delete(0, listData2.size() - 1)
    listData3.delete(0, listData3.size() - 1)
    listData4.delete(0, listData4.size() - 1)
    listData5.delete(0, listData5.size() - 1)

    for item1, item2, item3, item4, item5 in zip(strData1,strData2,strData3,strData4,strData5) :
        listData1.insert(END, item1);   listData2.insert(END,item2);
        listData3.insert(END, item3);   listData4.insert(END,item4);
        listData5.insert(END, item5);
    conn.close()

def imageSelectData():
    global conn, cur, data1, data2, data3, data4,data5,sql
    strData1, strData2, strData3, strData4, strData5 = [],[],[],[],[]
    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()
    data1 = edt1.get()

    sql = "SELECT * FROM member_table JOIN photo_table ON member_table.m_id = photo_table.p_upUser "
    sql += "WHERE member_table.m_id = '" + data1 + "'"
    cur.execute(sql)
    strData1.append("                 ID")
    strData2.append("              파일명")
    strData3.append("            파일 크기")
    strData4.append("          이미지 크기")
    strData5.append("               시간");
    strData1.append("=" * 20);strData2.append("=" * 20);strData3.append("=" * 20);
    strData4.append("=" * 20);strData5.append("=" * 20);
    while (True) :
        row = cur.fetchone()
        if row == None:
            break;
        strData1.append(row[0]); strData2.append(str(row[6])+'.'+str(row[7])); strData3.append(row[8]);
        strData4.append(str(row[9])+ 'x' + str(row[10])); strData5.append(row[12]);

    listData1.delete(0, listData1.size() - 1)
    listData2.delete(0, listData2.size() - 1)
    listData3.delete(0, listData3.size() - 1)
    listData4.delete(0, listData4.size() - 1)
    listData5.delete(0, listData5.size() - 1)

    for item1, item2, item3, item4, item5 in zip(strData1,strData2,strData3,strData4,strData5) :
        listData1.insert(END, item1);   listData2.insert(END,item2);
        listData3.insert(END, item3);   listData4.insert(END,item4);
        listData5.insert(END, item5);
    conn.close()

##메인 코드부
def main(login) :
    global window, edtFrame,listFrame, edt1, edt2, edt3, edt4, edt5
    global listData1,listData2,listData3,listData4,listData5
    window = Tk()
    window.title('관리자 모드')
    window.geometry('800x400')

    ### 메뉴 만들기 ###
    edtFrame = Frame(window, bg='#505050')
    edtFrame.pack(fill=BOTH)
    listFrame = Frame(window)
    listFrame.pack(side = BOTTOM, fill=BOTH, expand = 1)


    v1 = StringVar(window, value='ID')
    v2 = StringVar(window, value='Password')
    v3 = StringVar(window, value='이름')
    v4 = StringVar(window, value='전화번호')
    v5 = StringVar(window, value='이메일')



    edt1 = Entry(edtFrame, textvariable = v1, width=10);    edt1.pack(side=LEFT, padx=10, pady=10)
    edt2 = Entry(edtFrame, textvariable = v2, width=12);    edt2.pack(side=LEFT, padx=10, pady=10)
    edt3 = Entry(edtFrame, textvariable = v3, width=10);    edt3.pack(side=LEFT, padx=10, pady=10)
    edt4 = Entry(edtFrame, textvariable = v4, width=10);    edt4.pack(side=LEFT, padx=10, pady=10)
    edt5 = Entry(edtFrame, textvariable = v5, width=12);    edt5.pack(side=LEFT, padx=10, pady=10)


    btnInsert = Button(edtFrame, text="입력", command=insertData)
    btnInsert.pack(side=LEFT, padx=10, pady=10)
    btnDelete = Button(edtFrame, text="삭제", command=deleteData)
    btnDelete.pack(side=LEFT, padx=10, pady=10)
    btnEdit = Button(edtFrame, text="수정", command=editData)
    btnEdit.pack(side=LEFT, padx=10, pady=10)
    btnSelect = Button(edtFrame, text="조회", command=selectData)
    btnSelect.pack(side=LEFT, padx=10, pady=10)
    btnImageSelect = Button(edtFrame, text="영상 조회", command=imageSelectData)
    btnImageSelect.pack(side=LEFT, padx=10, pady=10)


    listData1 = Listbox(listFrame, bg = 'white');
    listData1.pack(side=LEFT, fill=BOTH, expand=1)
    listData2 = Listbox(listFrame, bg='white');
    listData2.pack(side=LEFT, fill=BOTH, expand=1)
    listData3 = Listbox(listFrame, bg='white');
    listData3.pack(side=LEFT, fill=BOTH, expand=1)
    listData4 = Listbox(listFrame, bg='white');
    listData4.pack(side=LEFT, fill=BOTH, expand=1)
    listData5 = Listbox(listFrame, bg='white');
    listData5.pack(side=LEFT, fill=BOTH, expand=1)

    ######################

    window.mainloop()