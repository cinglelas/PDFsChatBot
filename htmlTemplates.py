css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://media.licdn.com/dms/image/C5603AQG7MJisfvrE3A/profile-displayphoto-shrink_400_400/0/1628494182857?e=1713398400&v=beta&t=enRXLRaQZ-K0zjCaBTSMx8S-ziQ8qGqyBq4_oZp7SOM" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://media.licdn.com/dms/image/D5603AQHM0nRnFXlggg/profile-displayphoto-shrink_400_400/0/1663719547520?e=1713398400&v=beta&t=M9UlR0r-gRQZs65vBir9DlnXVk02C3__mh-bGa0yQhA">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''