from app.processing.EmoSentinel_processing import emo_sentinel_train,emo_sentinel_base_detect

def post_emo_sentinel_train(model_details):
    print(model_details['model_name'])
    print(model_details['batch_size'])
    print(model_details['epochs'])
    return emo_sentinel_train(model_details)

def post_emo_sentinel_base_detect(body):
    return emo_sentinel_base_detect(body['text'])