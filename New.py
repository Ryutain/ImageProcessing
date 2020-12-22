import pymysql
from tkinter import *
from tkinter import messagebox

##전역 변수부
#DB 관련
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
        messagebox.showerror('오류', 'id 입력이 잘못되었습니다.')
    else:
        messagebox.showinfo('성공', '회원 가입이 완료되었습니다.')
        window.destroy()
    conn.commit()
    conn.close()

def keyEvent(event) :
    if event.keysym == "Return" :
        insertData()



##메인 코드부
def main() :
    global window, edt1, edt2, edt3, edt4, edt5
    global listData1,listData2,listData3,listData4,listData5
    window = Tk()
    window.title('회원 가입')
    window.geometry('200x350')
    window.bind("<Key>", keyEvent)

    ### 메뉴 만들기 ###
    v1 = StringVar(window, value='ID')
    v2 = StringVar(window, value='PASSWORD')
    v3 = StringVar(window, value='이름')
    v4 = StringVar(window, value='전화번호')
    v5 = StringVar(window, value='이메일')

    canvas = Canvas(window, width=350, height=200, bg='#E0FFFF')
    canvas.pack(expand=YES, fill=BOTH)

    edt1 = Entry(window, textvariable=v1, width=20);
    edt1.pack()
    edt2 = Entry(window, textvariable=v2, width=20);
    edt2.pack()
    edt3 = Entry(window, textvariable=v3, width=20);
    edt3.pack()
    edt4 = Entry(window, textvariable=v4, width=20);
    edt4.pack()
    edt5 = Entry(window, textvariable=v5, width=20);
    edt5.pack()
    photo = PhotoImage(file='join.PNG',master=window)
    label1 = Label(window, image=photo)
    label1.pack()


    btnInsert = Button(window, text="가입하기", font=('맑은 고딕체', 25, 'bold'),
                       width = 7, bg='#505050', fg='white', command=insertData)
    btnInsert.pack(side=LEFT, padx=10, pady=10)

    label1.place(x=0, y=0);
    edt1.place(x=25, y=70);
    edt2.place(x=25, y=105);
    edt3.place(x=25, y=140);
    edt4.place(x=25, y=175);
    edt5.place(x=25, y=210);
    btnInsert.place(x=22, y=250);

    ######################

    window.mainloop()