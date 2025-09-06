<script setup lang="ts">
import { ProgressSpinner, Panel, Button } from 'primevue'
import FileUpload, { type FileUploadUploadEvent } from 'primevue/fileupload'
import { ref } from 'vue'

enum State {
  Idle,
  Uploading,
  Success,
  Error,
}

const response = ref({ mark: 0, feedback: '' })

const state = ref(State.Idle)

const upload = (a: FileUploadUploadEvent) => {
  response.value = JSON.parse(a!.xhr.response)
  state.value = State.Success
}
</script>

<template>
  <main>
    <form action="http://127.0.0.1:8000/feedback/" method="POST" enctype="multipart/form-data">
      <p>
        <label for="piece_1_name">Piece 1:</label>
        <input type="text" id="piece_1_name" name="piece_1_name" />
        <input type="file" id="piece_1_audio" name="piece_1_audio" accept="audio/mp3" />
      </p>

      <p>
        <label for="piece_2_name">Piece 2:</label>
        <input type="text" id="piece_2_name" name="piece_2_name" />
        <input type="file" id="piece_2_audio" name="piece_2_audio" accept="audio/mp3" />
      </p>

      <input type="submit" value="Submit" />
    </form>

    <FileUpload
      v-if="state === State.Idle"
      @upload="upload"
      name="demo[]"
      url="http://127.0.0.1:8000/feedback/"
      :multiple="true"
      accept="audio/*"
    >
      <template #empty>
        <span>Drag and drop files to here to upload.</span>
      </template>
    </FileUpload>
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
      <h2>Mark: {{ response.mark }}/100</h2>
      <p>{{ response.feedback }}</p>
    </Panel>
  </main>
</template>
