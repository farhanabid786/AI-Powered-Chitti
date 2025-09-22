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

# css = '''
# <style>
# body {
#     background-color: #1e1e2f;
#     font-family: 'Segoe UI', sans-serif;
# }
# .chat-message {
#     padding: 1rem;
#     border-radius: 0.75rem;
#     margin-bottom: 1rem;
#     display: flex;
#     align-items: flex-start;
#     box-shadow: 0 4px 12px rgba(0,0,0,0.25);
#     transition: transform 0.2s ease, background-color 0.3s ease;
# }
# .chat-message.user {
#     background: linear-gradient(135deg, #2b313e, #3a3f4b);
# }
# .chat-message.bot {
#     background: linear-gradient(135deg, #475063, #5a6277);
# }
# .chat-message:hover {
#     transform: scale(1.02);
# }
# .chat-message .avatar {
#     width: 12%;
#     text-align: center;
# }
# .chat-message .avatar img {
#     max-width: 55px;
#     max-height: 55px;
#     border-radius: 50%;
#     object-fit: cover;
#     border: 2px solid #ffffff;
# }
# .chat-message .message {
#     width: 85%;
#     padding: 0 1rem;
#     color: #ffffff;
#     font-size: 1rem;
#     word-wrap: break-word;
#     line-height: 1.5;
# }
# .timestamp {
#     display: block;
#     font-size: 0.75rem;
#     color: #cfcfcf;
#     margin-top: 0.25rem;
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
#     padding: 0.6rem 1rem;
#     border: none;
#     font-weight: bold;
#     transition: background 0.3s ease;
# }
# .stButton>button:hover {
#     background-color: #357ABD;
# }
# </style>
# '''

# bot_template = '''
# <div class="chat-message bot">
#     <div class="avatar">
#         <img src="https://i.postimg.cc/cJ4tvzrT/46387796-274498329873623-253092286630461440-n.jpg" alt="bot">
#     </div>
#     <div class="message">
#         {{MSG}}
#         <span class="timestamp">{{TIME}}</span>
#     </div>
# </div>
# '''

# user_template = '''
# <div class="chat-message user">
#     <div class="avatar">
#         <img src="https://i.postimg.cc/HsDmZ74C/1754987630509.jpg" alt="user">
#     </div>
#     <div class="message">
#         {{MSG}}
#         <span class="timestamp">{{TIME}}</span>
#     </div>
# </div>
# '''

css = '''
<style>
/* Background */
body {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #f1f1f1;
}

/* Title + subtitle styling */
h1, h2, h3, .stSubheader, .stHeader {
    text-align: center;
    background: linear-gradient(90deg, #ff9966, #ff5e62);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    letter-spacing: 1px;
}

/* Chat container */
.chat-message {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 1rem;
    padding: 1rem 1.2rem;
    border-radius: 1rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

/* Hover glow */
.chat-message:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 25px rgba(255,255,255,0.15);
}

/* User message bubble */
.chat-message.user {
    background: linear-gradient(135deg, rgba(52, 152, 219, 0.2), rgba(41, 128, 185, 0.4));
    border-left: 4px solid #3498db;
}

/* Bot message bubble */
.chat-message.bot {
    background: linear-gradient(135deg, rgba(155, 89, 182, 0.2), rgba(142, 68, 173, 0.4));
    border-left: 4px solid #9b59b6;
}

/* Avatar */
.chat-message .avatar img {
    max-width: 55px;
    max-height: 55px;
    border-radius: 50%;
    border: 2px solid #ffffff;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.6);
    object-fit: cover;
}

/* Message text */
.chat-message .message {
    flex: 1;
    font-size: 1.05rem;
    line-height: 1.6;
    color: #fdfdfd;
    font-weight: 400;
    word-wrap: break-word;
    letter-spacing: 0.3px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2c3e50, #1e1e2f);
    color: white;
}
</style>
'''

# Bot template
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.postimg.cc/cJ4tvzrT/46387796-274498329873623-253092286630461440-n.jpg" alt="Bot Avatar">
    </div>
    <div class="message">
        {{MSG}}
    </div>
</div>
'''

# User template
user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.postimg.cc/HsDmZ74C/1754987630509.jpg" alt="User Avatar">
    </div>    
    <div class="message">
        {{MSG}}
    </div>
</div>
'''


