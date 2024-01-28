import os
import ssl
import smtplib


from email.message import EmailMessage

if os.path.isfile('env.py'):
    import env


def send_email(to_email, msg, subject):

    # Define email sender and receiver
    email_sender = 'sunnylaw18@gmail.com'
    email_password = os.environ.get('GMAIL_PASSWORD2')
    email_receiver = to_email

    # Set the subject and body of the email

    body = "Thank you for using Market Tracker!\n" + msg

    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = to_email
    em['Subject'] = subject
    em.set_content(body)

    # Add SSL (layer of security)
    context = ssl.create_default_context()

    # Log in and send email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())