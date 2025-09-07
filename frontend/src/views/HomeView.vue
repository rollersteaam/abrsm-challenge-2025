<script setup lang="ts">
import { ProgressSpinner, Panel, Button } from 'primevue'
import { ref } from 'vue'

enum State {
  Idle,
  Uploading,
  Success,
  Error,
}

const data = ref({ mark: 0, feedback: '' })

const state = ref(State.Idle)

// const upload = (a: FileUploadUploadEvent) => {
//   response.value = JSON.parse(a!.xhr.response)
//   state.value = State.Success
// }

// async function onFileUpload(event: FileUploadUploaderEvent) {
//   const files = event.files
//   const formData = new FormData()

//   // Preview and formData population
//   for (let i = 0; i < (files as File[]).length; i++) {
//     const file = (files as File[])[i]
//     const key = `piece_${i + 1}_name`
//     formData.append(key, file)

//     // For preview
//     // const reader = new FileReader()
//     // reader.onload = (e) => {
//     //   srcList.value.push(e.target.result)
//     // }
//     // reader.readAsDataURL(file)
//   }
//   console.log({ formData })
//   try {
//     const response = await fetch('http://127.0.0.1:8000/feedback/', {
//       method: 'POST',
//       body: formData,
//     })
//     data.value = await response.json()
//   } catch (err) {
//     console.error('Upload failed:', err)
//   }
// }
// name="piece_1_audio"
// url="http://127.0.0.1:8000/feedback/"

const call = async (e: SubmitEvent) => {
  e.preventDefault()
  state.value = State.Uploading
  const formData = new FormData(e.target as HTMLFormElement)
  try {
    const response = await fetch('/feedback/', {
      method: 'POST',
      body: formData,
    })
    data.value = await response.json()
    state.value = State.Success
  } catch (err) {
    console.error('Upload failed:', err)
    state.value = State.Error
  }
}
</script>

<template>
  <main>
    <form v-if="state === State.Idle" @submit="call" enctype="multipart/form-data">
      <p>
        <label for="piece_1_name">Piece 1: </label>
        <!-- <input type="text" id="piece_1_name" name="piece_1_name" /> -->
        <input type="file" id="piece_1_audio" name="piece_1_audio" accept="audio/mp3" />
      </p>

      <p>
        <label for="piece_2_name">Piece 2: </label>
        <!-- <input type="text" id="piece_2_name" name="piece_2_name" /> -->
        <input type="file" id="piece_2_audio" name="piece_2_audio" accept="audio/mp3" />
      </p>

      <input type="submit" value="Submit" />
    </form>
    <ProgressSpinner v-if="state === State.Uploading"></ProgressSpinner>
    <Panel v-if="state === State.Success">
      <Button
        style="position: absolute; top: 2rem; right: 2rem"
        @click="state = State.Idle"
        icon="pi pi-times"
        variant="text"
        rounded
        aria-label="Cancel"
      />
      <h2>Mark: {{ data.mark }}/100</h2>
      <p>{{ data.feedback }}</p>
    </Panel>
  </main>
</template>
