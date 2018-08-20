# -*- coding: utf-8 -*-

# These emojis are described in the Appendix of the paper.

import re
from twokenize import simpleTokenize

MENTION = '_TWITTER-ENTITY_'

# emojis from: https://github.com/wooorm/gemoji/blob/master/support.md
happy = list({':\)', ':-\)', ': \)', ':D', '=\)', ':-\)', '\(:', '\(-:', '\(=',
              '\\\ud83d\\\ude00', '\\\ud83d\\\ude03', '\\\ud83d\\\ude04', '\\\ud83d\\\ude01', '\\\ud83d\\\ude06',
              '\\\ud83d\\\ude05', '\\\ud83d\\\ude02', '\\\ud83e\\\udd23', '\\\u263a\\\ufe0f', '\\\ud83d\\\ude0a',
              '\\\ud83d\\\ude42', '\\\ud83d\\\ude0d', '\\\ud83d\\\ude18', '\\\ud83d\\\ude1c', '\\\ud83d\\\ude1d',
              '\\\ud83d\\\ude08', '\\\ud83d\\\ude39', '\\\ud83d\\\ude3a'})
# 5 regular happy smileys,
# emojis: grinning, smiley, smile, grin, laughing; satisfied,
# sweat_smile, joy, rofl, relaxed, blush, slightly_smiling_face,
# heart_eyes, kissing_heart, stuck_out_tongue_winking_eye, stuck_out_tongue_closed_eyes
# smiling_imp, joy_cat, smiley_cat
sad = list({':\(', ':-\(', ': \(', ':-\(', '=\(', '\):', '\)-:', '\) :', '\)=',
            # ':\\', ':/',
            '\\\ud83d\\\ude12', '\\\ud83d\\\ude1e',
            '\\\ud83d\\\ude14', '\\\u2639\\\ufe0f', '\\\ud83d\\\ude23', '\\\ud83d\\\ude2b', '\\\ud83d\\\ude29',
            '\\\ud83d\\\ude24', '\\\ud83d\\\ude20', '\\\ud83d\\\ude21', '\\\ud83d\\\ude22', '\\\ud83d\ude2d',
            '\\\ud83d\\\ude28', '\\\ud83d\\\ude30', '\\\ud83e\\\udd12', '\\\ud83d\\\udc7f', '\\\ud83e\\\udd22',
            '\\\ud83d\\\ude1f', '\\\ud83d\\\ude41', '\\\ud83d\\\ude16', '\\\ud83d\\\ude26', '\\\ud83d\\\ude27',
            '\\\ud83d\\\ude3f'})
# 3 regular sad smileys
# emojis: unamused, disappointed, pensive, frowning_face, persevere, tired_face,
# weary, triumph, angry, rage, cry, sob, fearful, cold_sweat, face_with_thermometer,
# imp, nauseated_face, worried, slightly_frowning_face, confounded, frowning, anguished
# crying_cat_face


# credit to SO answers: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
emoji_pattern = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\ud83c[\udde0-\uddff])|"  # flags (iOS)
    u"(:\)+)|"
    u"(:\(+)|"
    u"(:p)|"
    u"(:D)|"
    u"(;\)+)|"
    u"(=\)+)|"
    u"(=\(+)|"
    u"(:-\)+)|"
    u"(:-\(+)|"

    u"(\(+:)|"
    u"(\)+:)|"
    u"(\(+=)|"
    u"(\)+=)|"
    u"(\(+-:)|"
    u"(\)+-:)"
    "+", flags=re.UNICODE)


def remove_emojis(text):
    str = text.decode('unicode_escape')
    no_emojis_txt = emoji_pattern.sub('', str)
    no_emojis_txt = no_emojis_txt.encode('utf-8').strip()
    return no_emojis_txt


def normalize_text(text):
    no_emojs = remove_emojis(text)
    if len(no_emojs) == 0:
        return []
    toks = simpleTokenize(no_emojs)
    norm = []
    for t in toks:
        t = t.replace('\n', '')
        if t.startswith('@'):
            norm.append(MENTION)
        else:
            norm.append(t)
    return norm
