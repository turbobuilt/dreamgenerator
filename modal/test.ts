import axios from "axios";

export async function main() {
    let url = "https://noise-destroyer--stable-diffusion-xl-app-turbobuilt-dev.modal.run/infer";
    let response = await axios.post(url, {
        prompt: "a dog",
        task_id: "a12345",
        upload_url: "None",
        progress_url: "None"
    });
}