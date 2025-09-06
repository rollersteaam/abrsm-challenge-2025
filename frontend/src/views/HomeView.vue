<script setup lang="ts">

</script>

<template>
  <main>
    <!-- type="multipart/form-data" -->
    <form action="http://127.0.0.1:8000/feedback/" method="POST" enctype="multipart/form-data">
      <p>
        <label for="piece_1_name">Piece 1:</label>
        <input type="text" id="piece_1_name" name="piece_1_name">
        <input type="file" id="piece_1_audio" name="piece_1_audio" accept="audio/mp3" />
      </p>

      <p>
        <label for="piece_2_name">Piece 2:</label>
        <input type="text" id="piece_2_name" name="piece_2_name"></input>
        <input type="file" id="piece_2_audio" name="piece_2_audio" accept="audio/mp3" />
      </p>

      <input type="submit" value="Submit">
    </form>
  </main>
</template>
