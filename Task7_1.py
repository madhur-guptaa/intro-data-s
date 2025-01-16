import getpass

import pymysql
from sshtunnel import SSHTunnelForwarder

# Get SSH username and password (Studium username and password)
ssh_username = input("Enter your SSH username: ")
ssh_password = getpass.getpass("Enter your SSH password: ")
db_user = 'ht24_2_group_13'
db_password = 'pasSWd_13'
# Set SSH tunnel
tunnel = SSHTunnelForwarder(
    ('fries.it.uu.se', 22),  # port = 22
    ssh_username=ssh_username,
    ssh_password=ssh_password,
    remote_bind_address=('127.0.0.1', 3306)
)
# Start SSH tunnel
tunnel.start()
# Connect to MySQL database
mydb = pymysql.connect(
    host='127.0.0.1',
    user=db_user,
    password=db_password,
    port=tunnel.local_bind_port,
    db='ht24_2_project_group_13'
)
# Excute MySQL statements
mycursor = mydb.cursor()
department_id = input("Please enter the DEPARTMENT ID:")
mycursor.execute(f'''select Departments.department_path from Departments where department_id = '{department_id}'
''')
departments = []
for row in mycursor:

    if isinstance(row[0], str):
        departments = row[0].split('/')
        print("Split departments:", departments)

leaf = None
if len(departments) > 1:
    leaf = departments[-1]
    mycursor.execute(f'''select P.product_id,
       P.title,
       P.short_description,
       (P.price * (1 + (P.tax / 100))) * ((100 - P.discount) / 100)
from Products P
where P.department_id = {department_id}
    ''')

    leaf = mycursor.fetchall()
else:
    mycursor.execute(f'''select Departments.title, Departments.department_id
from Departments
where parent_id = {department_id}
        ''')
    leaf = mycursor.fetchall()

print(leaf)

mydb.close()
tunnel.stop()
