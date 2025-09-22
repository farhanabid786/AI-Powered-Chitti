# css = '''
# <style>
# .chat-message {
#     padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
# }
# .chat-message.user {
#     background-color: #2b313e
# }
# .chat-message.bot {
#     background-color: #475063
# }
# .chat-message .avatar {
#   width: 20%;
# }
# .chat-message .avatar img {
#   max-width: 78px;
#   max-height: 78px;
#   border-radius: 50%;
#   object-fit: cover;
# }
# .chat-message .message {
#   width: 80%;
#   padding: 0 1.5rem;
#   color: #fff;
# }
# '''

# bot_template = '''
# <div class="chat-message bot">
#     <div class="avatar">
#         <img src="https://i.postimg.cc/cJ4tvzrT/46387796-274498329873623-253092286630461440-n.jpg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
#     </div>
#     <div class="message">{{MSG}}</div>
# </div>
# '''

# user_template = '''
# <div class="chat-message user">
#     <div class="avatar">
#         <img src="https://i.postimg.cc/HsDmZ74C/1754987630509.jpg">
#     </div>    
#     <div class="message">{{MSG}}</div>
# </div>
# '''

# css = '''
# <style>
# body {
#     background-color: #1e1e2f;
#     font-family: 'Arial', sans-serif;
# }

# h1 {
#     color: #ffffff;
#     text-align: center;
#     margin-bottom: 2rem;
# }

# .chat-message {
#     padding: 1.5rem;
#     border-radius: 0.5rem;
#     margin-bottom: 1rem;
#     display: flex;
#     align-items: flex-start;
#     box-shadow: 0 4px 8px rgba(0,0,0,0.2);
#     transition: background-color 0.3s ease;
# }

# .chat-message.user {
#     background-color: #2b313e;
# }

# .chat-message.bot {
#     background-color: #475063;
# }

# .chat-message.user:hover {
#     background-color: #3a3f4b;
# }

# .chat-message.bot:hover {
#     background-color: #5a6277;
# }

# .chat-message .avatar {
#     width: 20%;
# }

# .chat-message .avatar img {
#     max-width: 78px;
#     max-height: 78px;
#     border-radius: 50%;
#     object-fit: cover;
#     border: 2px solid #ffffff;
# }

# .chat-message .message {
#     width: 80%;
#     padding: 0 1.5rem;
#     color: #ffffff;
#     font-size: 1rem;
#     word-wrap: break-word;
# }

# .stTextInput>div>div>input {
#     background-color: #2b313e;
#     color: #ffffff;
#     border: none;
#     border-radius: 0.5rem;
#     padding: 0.75rem;
# }

# .stButton>button {
#     background-color: #4a90e2;
#     color: white;
#     border-radius: 0.5rem;
#     padding: 0.5rem 1rem;
#     border: none;
#     font-weight: bold;
# }

# .stButton>button:hover {
#     background-color: #357ABD;
# }
# </style>
# '''

# bot_template = '''
# <div class="chat-message bot">
#     <div class="avatar">
#         <img src="https://i.postimg.cc/cJ4tvzrT/46387796-274498329873623-253092286630461440-n.jpg" alt="Bot Avatar">
#     </div>
#     <div class="message">{{MSG}}</div>
# </div>
# '''

# user_template = '''
# <div class="chat-message user">
#     <div class="avatar">
#         <img src="https://i.postimg.cc/HsDmZ74C/1754987630509.jpg" alt="User Avatar">
#     </div>    
#     <div class="message">{{MSG}}</div>
# </div>
# '''

css = '''
<style>
body {
    background-color: #1e1e2f;
    font-family: 'Segoe UI', sans-serif;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.75rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    transition: transform 0.2s ease, background-color 0.3s ease;
}
.chat-message.user {
    background: linear-gradient(135deg, #2b313e, #3a3f4b);
}
.chat-message.bot {
    background: linear-gradient(135deg, #475063, #5a6277);
}
.chat-message:hover {
    transform: scale(1.02);
}
.chat-message .avatar {
    width: 12%;
    text-align: center;
}
.chat-message .avatar img {
    max-width: 55px;
    max-height: 55px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid #ffffff;
}
.chat-message .message {
    width: 85%;
    padding: 0 1rem;
    color: #ffffff;
    font-size: 1rem;
    word-wrap: break-word;
    line-height: 1.5;
}
.timestamp {
    display: block;
    font-size: 0.75rem;
    color: #cfcfcf;
    margin-top: 0.25rem;
}
.stTextInput>div>div>input {
    background-color: #2b313e;
    color: #ffffff;
    border: none;
    border-radius: 0.5rem;
    padding: 0.75rem;
}
.stButton>button {
    background-color: #4a90e2;
    color: white;
    border-radius: 0.5rem;
    padding: 0.6rem 1rem;
    border: none;
    font-weight: bold;
    transition: background 0.3s ease;
}
.stButton>button:hover {
    background-color: #357ABD;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.postimg.cc/cJ4tvzrT/46387796-274498329873623-253092286630461440-n.jpg" alt="bot">
    </div>
    <div class="message">
        {{MSG}}
        <span class="timestamp">{{TIME}}</span>
    </div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.postimg.cc/HsDmZ74C/1754987630509.jpg" alt="user">
    </div>
    <div class="message">
        {{MSG}}
        <span class="timestamp">{{TIME}}</span>
    </div>
</div>
'''

