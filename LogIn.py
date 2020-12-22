import ImageProcessing
import Admin
import New
import pymysql

from tkinter import *
from tkinter import messagebox

def Login() :
    global conn,cur,sql,IP,USER,PASSWORD,DB
    login = IDedt.get()
    password = PASSedt.get()
    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()
    sql = "SELECT m_pass FROM member_table WHERE m_id='" + login + "'"
    cur.execute(sql)
    row = cur.fetchone()
    if login == 'root' :
        if password == '1234':
            window.destroy()
            Admin.main(login)
        else :
            messagebox.showerror('오류', '패스워드가 틀렸습니다.')
    else :
        if password == row[0] :
            window.destroy()
            ImageProcessing.main(login)
        else :
            messagebox.showerror('오류', '패스워드가 틀렸습니다.')

def Join() :
    New.main()

def keyEvent(event) :
    if event.keysym == "Return" :
        Login()

##전역 변수부
#DB 관련
conn, cur = None, None

IP = '127.0.0.1'
USER = 'root'
PASSWORD = '1234'
DB = 'photo_db'
sql = ""


##메인 함수부
window = Tk()
window.title("회원 로그인")
window.geometry('400x300')
window.resizable(width=False, height=False)
window.bind("<Key>", keyEvent)
canvas = Canvas(window, width=300, height=400, bg='#E0FFFF')
canvas.pack(expand=YES, fill=BOTH)
v1 = StringVar(window, value='ID')
v2 = StringVar(window, value='PASSWORD')
IDedt = Entry(window, textvariable = v1, width = 20)
PASSedt = Entry(window, textvariable = v2, width = 20)
loginbtn = Button(window, text='로그인', font=('맑은 고딕체', 12, 'bold'),width = 7, bg='#B22222', fg='white', command=Login)
joinbtn = Button(window, text='회원가입', font=('맑은 고딕체', 12, 'bold'),width = 7, bg='#008B8B', fg='white', command=Join)
photo = PhotoImage(file='login.png',master=window)
label3 = Label(window, image=photo)

IDedt.pack();IDedt.place(x=50, y=180)
PASSedt.pack();PASSedt.place(x=50, y=220)
loginbtn.pack();loginbtn.place(x= 240, y=170)
joinbtn.pack();joinbtn.place(x= 240, y=220)
label3.pack(); label3.place(x=0, y = 0)


window.mainloop()
