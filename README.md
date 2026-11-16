# ערכת התחלה: מערכת RAG לחברת הרכב "אופק רכב"

החברה והנתונים דמיוניים, ונכתבו לצורך התרגיל.

## מה יש בתיקייה

- `data/` — 16 מסמכים של החברה: מפרט טכני לכל דגם, מחירון, קטלוג אביזרים, אחריות, טיפולים, קודי התרעה וקריאות שירות.
- `golden.json` — 15 שאלות בדיקה, ולכל שאלה הטקסט שחייב להופיע בקטע הנכון.
- `rag/` — הצינור: `loader.py`, `chunking.py`, `embed.py`, `search.py`, `answer.py`.
- `eval_retrieval.py` — מדידת האחזור. חינם, בלי קריאה למודל שפה.
- `eval_answers.py` — מדידת התשובות דרך OpenAI. עולה בערך סנט אחד עם gpt-4o-mini (אפשר להחליף מודל עם `LLM_MODEL`).
- `.cache/` — וקטורים שכבר חושבו מראש, כדי שההרצה הראשונה תהיה מהירה.

## התקנה

נדרש Python 3.10 ומעלה.

```
pip install -r requirements.txt
python eval_retrieval.py
```

אם הכול תקין, השורה האחרונה תהיה `total 6/13 0.37`.

## המקומות שבהם כותבים קוד

כל אחד מהם מסומן ב־`TODO`, ומעלה שגיאה ברורה עד שכותבים אותו:

- `rag/chunking.py` — `chunk_by_headings` (שלב 2)
- `rag/search.py` — `rrf` (שלב 3) ו־`is_current` (שלב 4)
- `rag/answer.py` — `verify_citations` (שלב 5)

## וקטורים

ברירת המחדל היא המודל `intfloat/multilingual-e5-small`, שרץ על המחשב בחינם ותומך בעברית. אפשר במקומו להשתמש ב־Voyage AI:

```
pip install voyageai
```

ואז להגדיר את משתני הסביבה `EMBED_BACKEND=voyage` ו־`VOYAGE_API_KEY`.

## מפתחות

`OPENAI_API_KEY` ו־`VOYAGE_API_KEY` מוגדרים רק כמשתני סביבה בטרמינל. לעולם לא בקוד ולא ב־git.
