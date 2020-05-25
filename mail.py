import smtplib
import urllib.request as urllib
# Senders email
sender_email = "doniksingh@gmail.com"
# Receivers email
rec_email = "adarsh.adarshsingh12@gmail.com"

message = "Hey Adarsh,Best Model has been created... Thank You"
# Initialize the server variable
server = smtplib.SMTP('smtp.gmail.com', 587)
# Start the server connection
server.starttls()
# Login
server.login("doniksingh@gmail.com", "PaSS@1234")
print("Login Success!")
# Send Email
server.sendmail("Adarsh singh", "adarsh.adarshsingh12@gmail.com", message)
print(f"Email has been sent successfully to {rec_email}")