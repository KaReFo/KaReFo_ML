import chainlit as cl
import ML as ml

@cl.on_message
async def main(message: str):
    # Your custom logic goes here...
    #ml.learningPhase()
    ml.isTrainedFlag = True
    output = ml.generatesingleoutput(message)
    # Send a response back to the user
    await cl.Message(
        content=f"Received: {output}",
    ).send()
