def recommend_music(sentiment):

    music_map = {
        "positive": {
            "genre": ["Pop", "Dance", "EDM"],
            "songs": [
                "Shape of You - Ed Sheeran",
                "Blinding Lights - The Weeknd",
                "Uptown Funk - Bruno Mars"
            ]
        },

        "neutral": {
            "genre": ["Jazz", "Classical"],
            "songs": [
                "Fly Me To The Moon",
                "Autumn Leaves",
                "Moonlight Sonata"
            ]
        },

        "negative": {
            "genre": ["Lo-fi", "Acoustic"],
            "songs": [
                "Someone Like You - Adele",
                "Let Her Go - Passenger",
                "Fix You - Coldplay"
            ]
        }
    }

    return music_map.get(sentiment)