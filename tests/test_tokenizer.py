from models.tokenizer import CloneHeroTokenizer

# Crear el tokenizador
tokenizer = CloneHeroTokenizer()

# Crear el tokenizer con el token Normal
tokenizer = CloneHeroTokenizer()

# Ejemplo con nota normal
normal_note_text = "<Guitar> <Expert> <Minute_3> <Beat_1> <Beatshift_1> <Normal> <Pitch_3>"
encoded_normal = tokenizer(normal_note_text, return_tensors='pt')

# Usar el método actualizado para codificar charts completos
chart_sequence = tokenizer.encode_complete_chart(
    instrument='<Guitar>',
    difficulty='<Expert>',
    duration_minutes=3,
    beat_sequence=[
        (1, 1, 3, 'normal'),   # beat, beatshift, pitch, note_type
        (1, 16, 5, 'normal'),
        (2, 1, 0, 'normal')
    ]
)

print(tokenizer.decode(chart_sequence, skip_special_tokens=False))
# Para drums con notas normales
drums_sequence = tokenizer.encode_complete_chart(
    instrument='<Drums>',
    difficulty='<Expert>',
    duration_minutes=3,
    beat_sequence=[
        (1, 1, 0, 'normal'),   # kick drum normal
        (1, 8, 2, 'normal'),   # snare normal
        (2, 1, 4, 'normal')    # crash normal
    ]
)

print(tokenizer.decode(chart_sequence, skip_special_tokens=False))