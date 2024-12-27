import asyncio

import numpy as np
import tritonclient.grpc.aio as grpcclient


async def try_request():
    url = "localhost:8001"  # host and port number of the triton inference server
    client = grpcclient.InferenceServerClient(url=url, verbose=False)

    model_name = "vllm_model"

    def create_request(prompt, request_id):
        input = grpcclient.InferInput("text_input", [1], "BYTES")
        prompt_data = np.array([prompt.encode("utf-8")])
        input.set_data_from_numpy(prompt_data)

        stream_setting = grpcclient.InferInput("stream", [1], "BOOL")
        stream_setting.set_data_from_numpy(np.array([True]))

        inputs = [input, stream_setting]

        output = grpcclient.InferRequestedOutput("text_output")
        outputs = [output]

        return {
            "model_name": model_name,
            "inputs": inputs,
            "outputs": outputs,
            "request_id": request_id,
        }

    prompt = "What is the first letter of English alphabets?"

    async def requests_gen():
        yield create_request(prompt, "req-0")

    response_stream = client.stream_infer(requests_gen())

    async for response in response_stream:
        result, error = response
        if error:
            print("Error occurred!")
        else:
            output = result.as_numpy("text_output")
            for i in output:
                decoded = i.decode("utf-8")
                if decoded.startswith(prompt):
                    decoded = decoded[len(prompt) :]
                print(decoded, end="", flush=True)


asyncio.run(try_request())
