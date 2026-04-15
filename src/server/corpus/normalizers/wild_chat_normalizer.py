import config
from .dataset_normalizer import IDatasetNormalizer


class WildChatNormalizer(IDatasetNormalizer):
    def filter(self, row):
        return (
            row["language"] == "English" and
            row["toxic"] == False and
            row["redacted"] == False
        )
        # return "conversation" in row
        # return True


    def map(self, row):
        conversation = ""
        for entry in row["conversation"]:
            content = entry["content"]
            role = entry["role"]
            if role == "user":
                conversation += f"[{config.USER_NAME}] {content}\n"
            else:
                conversation += f"[{config.MODEL_NAME}] {content}\n"

        return {
            "source": "WildChat",
            "chunk": conversation,
            # "turn": row["turn"],
            # "openai_moderation": row["openai_moderation"],
            # "detoxify_moderation": row["detoxify_moderation"]
        }
    
        messages = []

        for msg in row["conversation"]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return {
            "messages": messages,
            "source": "WildChat"
        }
