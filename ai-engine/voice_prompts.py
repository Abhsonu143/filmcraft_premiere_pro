"""voice_prompts: what to read aloud when recording a voice for cloning.

Three sets (English, Hindi, Hinglish). Each line has a delivery direction, because a clone copies
not only the timbre but how the reference was spoken: a voice recorded only in a flat reading
tone will narrate everything flatly. The lines together cover:

  * every sound of the language: English vowels, diphthongs, th/v/w/r/l clusters; Hindi aspirated
    (ख घ छ झ ठ ढ थ ध फ भ), retroflex (ट ठ ड ढ ण ड़ ढ़), nasal (ँ ं ङ ञ ण), nukta sounds
    (क़ ख़ ग़ ज़ फ़), conjuncts (क्ष त्र ज्ञ श्र) and long/short vowels;
  * the deliveries a video needs: calm narration, excitement, curiosity (questions), seriousness,
    warmth, suspense, a strong hook, a call to action;
  * numbers, names and English words inside Hindi, as creators really speak.

About 6–8 minutes of reading per language. More (and natural talking) only helps.
"""

from __future__ import annotations

# (id, text, emotion, direction)
ENGLISH = [
    ("en01", "Hey everyone, welcome back to the channel. Today's video is one I've wanted to make for a really long time.", "warm", "Friendly, like greeting people you know."),
    ("en02", "The quick brown fox jumps over the lazy dog, while three thin thieves think through their thousand theories.", "neutral", "Clear and even, normal speed."),
    ("en03", "We woke up at four in the morning, packed the van, and drove north until the road simply ran out.", "story", "Storytelling, relaxed, paint the picture."),
    ("en04", "And then, out of nowhere, the clouds opened up and the whole valley was glowing gold!", "excited", "Excited, rising energy, smile while you say it."),
    ("en05", "Have you ever wondered why some places feel completely different the moment you arrive?", "curious", "A real question, curious, let the end go up."),
    ("en06", "This is important, so please listen carefully: never fly a drone near an airport.", "serious", "Serious and firm, slower, stress 'never'."),
    ("en07", "Honestly, I didn't expect it to be this beautiful. I just stood there for a minute, not saying anything.", "calm", "Soft and calm, almost thinking aloud."),
    ("en08", "Nobody tells you this, but the best shots usually happen in the five minutes after sunset.", "hook", "Confident hook, lean in, a little secretive."),
    ("en09", "It cost us two thousand four hundred and fifty rupees, about thirty dollars, for the whole weekend.", "neutral", "Matter-of-fact, clear numbers."),
    ("en10", "Wait. Did you hear that? Something was moving behind those trees.", "suspense", "Suspense: quieter, slower, pauses at the full stops."),
    ("en11", "Rivers, rocks, villages and very old railway bridges: we really did see everything on this trip.", "story", "Lively storytelling, list rhythm."),
    ("en12", "If you enjoyed this, hit subscribe and tell me in the comments where we should go next.", "cta", "Upbeat call to action, friendly."),
    ("en13", "Her voice was warm, the evening air was cool, and the view over the water was pure magic.", "warm", "Gentle and warm."),
    ("en14", "Look at that. Just look at it. That is why we climbed for six hours.", "excited", "Proud, impressed, emphasis on 'that'."),
    ("en15", "The engine sputtered, the wheels skidded on the wet gravel, and we stopped just inches from the edge.", "suspense", "Tense, faster, then slow down at the end."),
    ("en16", "Okay, so here's the plan: we film the sunrise, grab breakfast, and head to the waterfall by nine.", "neutral", "Casual and quick, like explaining to a friend."),
    ("en17", "What would you do if you only had one day left in this city?", "curious", "Thoughtful question."),
    ("en18", "I'm genuinely grateful for every single one of you who watches these videos. Thank you.", "warm", "Sincere and slow."),
    ("en19", "Measure twice, cut once; that rule saved my edit more times than I can count.", "neutral", "Clear, slight smile."),
    ("en20", "No way! It actually worked! I can't believe it actually worked!", "excited", "Very excited, almost laughing."),
    ("en21", "The village sits at nearly three thousand metres, where the air is thin and the nights are freezing.", "story", "Documentary narrator, calm authority."),
    ("en22", "Sometimes the journey matters more than the destination. This was one of those times.", "calm", "Reflective ending, slow."),
    ("en23", "Plenty of people asked which camera I use, so let's quickly go through my whole kit.", "neutral", "Casual, normal speed."),
    ("en24", "Usually, I'd measure the treasure with leisure, but this vision was a genuine pleasure.", "neutral", "Clear: covers the 'zh' sound."),
]

HINDI = [
    ("hi01", "नमस्ते दोस्तों, चैनल पर आपका स्वागत है। आज का वीडियो मेरे दिल के बहुत करीब है।", "warm", "दोस्ताना, जैसे जान-पहचान वालों से बात कर रहे हों।"),
    ("hi02", "सुबह चार बजे हम उठे, सामान बाँधा और पहाड़ों की ओर निकल पड़े।", "story", "कहानी सुनाने के अंदाज़ में, आराम से।"),
    ("hi03", "और अचानक बादल हटे, और पूरी घाटी सुनहरी रोशनी से जगमगा उठी!", "excited", "उत्साह से, आवाज़ में ऊर्जा बढ़ाते हुए।"),
    ("hi04", "क्या आपने कभी सोचा है कि कुछ जगहें पहुँचते ही इतनी अलग क्यों महसूस होती हैं?", "curious", "सच्चा सवाल, आख़िर में आवाज़ ऊपर।"),
    ("hi05", "यह बात बहुत ज़रूरी है, ध्यान से सुनिए: हवाई अड्डे के पास कभी भी ड्रोन मत उड़ाइए।", "serious", "गंभीर और दृढ़, धीरे, 'कभी भी' पर ज़ोर।"),
    ("hi06", "सच कहूँ तो मुझे उम्मीद नहीं थी कि यह इतना ख़ूबसूरत होगा। मैं बस एक मिनट चुपचाप खड़ा रह गया।", "calm", "धीमे और शांत, जैसे मन में सोच रहे हों।"),
    ("hi07", "यह बात कोई नहीं बताता, लेकिन सबसे अच्छे शॉट सूरज ढलने के ठीक बाद के पाँच मिनट में मिलते हैं।", "hook", "आत्मविश्वास से, थोड़ा राज़ खोलने वाले अंदाज़ में।"),
    ("hi08", "पूरे सफ़र का ख़र्च लगभग दो हज़ार चार सौ पचास रुपये आया, यानी हर दिन आठ सौ से भी कम।", "neutral", "साफ़-साफ़, अंक स्पष्ट बोलें।"),
    ("hi09", "रुको। क्या तुमने वह आवाज़ सुनी? उन पेड़ों के पीछे कुछ हिल रहा था।", "suspense", "रहस्य: धीमे, हल्की आवाज़, पूर्णविराम पर रुकें।"),
    ("hi10", "घने जंगल, ठंडी झील, ढलान पर बसे घर और भूरे पत्थरों के बीच झरने की धारा।", "story", "चित्र बनाते हुए, सूची की लय में।"),
    ("hi11", "अगर आपको यह वीडियो पसंद आया, तो चैनल को सब्सक्राइब कीजिए और कमेंट में बताइए कि अगली बार कहाँ चलें।", "cta", "ऊर्जा के साथ, दोस्ताना अपील।"),
    ("hi12", "क्षितिज पर ढलता सूरज, त्रिकोण जैसी चोटियाँ और ज्ञान की बातें करते हुए बूढ़े बाबा।", "neutral", "स्पष्ट उच्चारण: क्ष, त्र, ज्ञ।"),
    ("hi13", "देखो, ज़रा देखो इसे! इसी नज़ारे के लिए तो हम छह घंटे चढ़ाई करके आए थे।", "excited", "गर्व और ख़ुशी से, 'देखो' पर ज़ोर।"),
    ("hi14", "गाड़ी का इंजन अटका, पहिये गीली बजरी पर फिसले, और हम किनारे से बस कुछ इंच पहले रुके।", "suspense", "तनाव के साथ, तेज़, फिर आख़िर में धीमे।"),
    ("hi15", "ठीक है, तो प्लान यह है: पहले सूर्योदय शूट करेंगे, फिर नाश्ता, और नौ बजे तक झरने पर पहुँचेंगे।", "neutral", "आम बातचीत की तरह, थोड़ा तेज़।"),
    ("hi16", "अगर इस शहर में आपके पास सिर्फ़ एक दिन बचा हो, तो आप क्या करेंगे?", "curious", "सोचने वाला सवाल।"),
    ("hi17", "आप सब जो ये वीडियो देखते हैं, उन सबका मैं दिल से शुक्रगुज़ार हूँ। धन्यवाद।", "warm", "सच्चे मन से, धीरे।"),
    ("hi18", "ग़ज़ब! यह सच में काम कर गया! मुझे यक़ीन ही नहीं हो रहा!", "excited", "बहुत उत्साह से, हँसते हुए।"),
    ("hi19", "यह गाँव लगभग तीन हज़ार मीटर की ऊँचाई पर बसा है, जहाँ हवा पतली है और रातें बर्फ़ जैसी ठंडी।", "story", "डॉक्यूमेंट्री वाले अंदाज़ में, शांत अधिकार के साथ।"),
    ("hi20", "कभी-कभी मंज़िल से ज़्यादा सफ़र मायने रखता है। यह भी वैसा ही एक सफ़र था।", "calm", "विचारशील अंत, धीरे।"),
    ("hi21", "फ़ोन, फ़िल्टर, ट्राइपॉड और ढेर सारी बैटरियाँ, सब कुछ इस छोटे से बैग में आ गया।", "neutral", "सहज, 'फ़' साफ़।"),
    ("hi22", "उसने धीरे से कहा, घबराओ मत, ढलान पर संभलकर चलो, और भरोसा रखो।", "warm", "नरम, भरोसा दिलाते हुए।"),
    ("hi23", "बारिश की बूँदें, मिट्टी की ख़ुशबू और दूर कहीं बजती घंटियाँ, यही तो पहाड़ों का जादू है।", "calm", "भावुक, धीमे।"),
    ("hi24", "सवाल यह नहीं है कि हम वहाँ पहुँचे या नहीं, सवाल यह है कि हमने रास्ते में क्या सीखा।", "serious", "गंभीर, ठहराव के साथ।"),
]

HINGLISH = [
    ("mx01", "दोस्तों, आज का video थोड़ा different है, क्योंकि हम सीधे mountain की top पर drone उड़ाने वाले हैं।", "excited", "Natural Hinglish, जैसे आप असल में बोलते हैं।"),
    ("mx02", "Honestly, यह shot लेने में पूरे तीन घंटे लगे, लेकिन result देखकर लगा कि worth it था।", "story", "कहानी की तरह, आराम से।"),
    ("mx03", "अगर आप beginner हैं, तो पहले basic settings सीखिए, फिर slowly manual mode पर जाइए।", "neutral", "समझाने वाले अंदाज़ में।"),
    ("mx04", "Comment करके बताइए कि next video किस topic पर चाहिए, और channel को subscribe ज़रूर कीजिए।", "cta", "ऊर्जा के साथ।"),
    ("mx05", "Trust me, यह जगह photos में जितनी अच्छी दिखती है, असल में उससे दस गुना ज़्यादा beautiful है।", "hook", "भरोसे से, hook की तरह।"),
    ("mx06", "Budget breakdown सुनिए: hotel पंद्रह सौ, food आठ सौ, और travel लगभग दो हज़ार।", "neutral", "साफ़ अंक।"),
    ("mx07", "क्या आपको पता है कि ज़्यादातर लोग यह simple सी mistake करते हैं?", "curious", "सवाल, आख़िर में आवाज़ ऊपर।"),
    ("mx08", "Seriously, इतना peaceful moment मैंने बहुत time बाद feel किया।", "calm", "शांत और सच्चा।"),
]

SETS = {"en": ENGLISH, "hi": HINDI, "mix": HINGLISH}

FREE_TALK = {
    "en": "Now just talk for one minute, the way you do in your videos: what you filmed last, and why you liked it.",
    "hi": "अब बस एक मिनट अपने वीडियो वाले अंदाज़ में बात कीजिए: आख़िरी बार आपने क्या शूट किया और वह आपको क्यों पसंद आया।",
}


def prompts(languages: list[str]) -> list[dict]:
    """The reading list for the chosen languages (Hindi includes the Hinglish lines)."""
    out = []
    for lang in languages:
        for set_key in (["hi", "mix"] if lang == "hi" else [lang]):
            for pid, text, emotion, direction in SETS.get(set_key, []):
                out.append({"id": pid, "lang": "hi" if set_key == "mix" else set_key, "text": text, "emotion": emotion, "direction": direction})
        if lang in FREE_TALK:
            out.append({"id": f"{lang}-free", "lang": lang, "text": FREE_TALK[lang], "emotion": "natural", "direction": "free", "free": True})
    return out
