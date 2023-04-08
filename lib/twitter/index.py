# from langchain.agents import Tool, BaseTool
import requests


def get_user_handle(access_token):
    response = requests.get(
        "https://api.twitter.com/2/users/me",
        headers={
            "Authorization": f"Bearer {access_token}",
        }
    )

    if response.status_code == 200:
        data = response.json()
        return data["data"]['username']
    else:
        return None

def send_tweet(msg, access_token):
    try:
        user_handle = get_user_handle(access_token)

        if user_handle is None:
            return "Unable to get the user handle. Can't send Tweet."

        response = requests.post(
            "https://api.twitter.com/2/tweets",
            headers={
                "Authorization": f"Bearer {access_token}",
            },
            json={
                "status": msg
            }
        )
        
        data = response.json()

        print(data)
        
        if data['data']['id'] is not None:
            return f"Tweet sent successfully! Check it at https://twitter.com/{user_handle}/status/{data['data']['id']}."
        else:
            return "Tweet failed to send."
    except Exception as e:
        print("error")