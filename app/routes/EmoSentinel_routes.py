from fastapi import APIRouter, Query
from typing import Optional
from app.handlers.EmoSentinel_handlers import post_emo_sentinel_train,post_emo_sentinel_base_detect
emosentinel_router = APIRouter(prefix="/EmoSentinel/api/v1")


@emosentinel_router.get("/getparams/{model_name}")
async def get_model_params():
    return {"message": "This is FastAPI GET route 1"}

@emosentinel_router.post("/train")
async def post_emo_sentinel_train_route(model_details: dict):
    print(model_details)
    result =post_emo_sentinel_train(model_details)
    return {"message": "Training Completed", "model": model_details, "metrics":result }

@emosentinel_router.post("/basemodel/detect")
async def post_emo_sentinel_base_detect_route(body_details: dict):
    print("BODY::",body_details)
    result = post_emo_sentinel_base_detect(body_details)
    return{"text": body_details['text'], "flag" : result}


#@emosentinel_router.get("/route1")
#async def get_route1(id: int, name: Optional[str] = Query(None)):
    #return {"message": f"This is FastAPI GET route 1 with id: {id} and name: {name}"}

#@emosentinel_router.post("/route2/{item_id}")
#async def post_route2(item: dict):
    #return {"message": f"This is FastAPI POST route 2 with item_id", "item": item}
