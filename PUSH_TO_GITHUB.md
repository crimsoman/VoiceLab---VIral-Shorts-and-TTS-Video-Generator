# Pushing this to GitHub (simplest way)

You said you're logged into GitHub already but unsure of the rest — here's
the easiest path, no command-line typing needed.

## Method: GitHub Desktop (recommended for you)

1. Download **GitHub Desktop**: https://desktop.github.com/ and install it.
2. Open it and sign in with your GitHub account (the one you're already
   logged into on the website).
3. Click **File → Add Local Repository**.
4. Browse to and select your VoiceLab folder (the one with `server.py`,
   `index.html`, `setup.bat`, etc.).
5. It will say "This directory does not appear to be a Git repository" —
   click **"create a repository"** right there.
6. Fill in:
   - Name: `VoiceLab` (or whatever you like)
   - Leave everything else default
   - Click **Create Repository**
7. You'll now see all your files listed on the left as changes to commit.
   - At the bottom left, type a summary like `Initial commit`
   - Click **Commit to main**
8. Click **Publish repository** at the top.
   - Untick "Keep this code private" if you want it public, or leave it
     ticked to keep it private for now — you can change this later on
     GitHub's website under repo **Settings → Danger Zone → Change
     visibility**.
   - Click **Publish Repository**

Done — your repo is now live on your GitHub account. Any time you change
files later, GitHub Desktop will show them as changes; just repeat step 7
(commit) and click **Push origin** to update GitHub.

## Before you publish — double check

- [ ] Open `.gitignore` in this folder — it's already set up to exclude
  `venv/`, `hf_cache/`, `exports/`, and your database/settings files, so
  none of your personal generated content gets uploaded. Don't remove
  these lines.
- [ ] Do a quick search through `server.py` and `index.html` yourself for
  anything personal (API keys, your name in a comment, etc.) — I already
  scrubbed the two hardcoded paths and the hardcoded default AI model I
  found, but a second look from you never hurts.

## Alternative: command line (if you ever want it)

If you later get comfortable with a terminal, this is the equivalent,
run once inside the VoiceLab folder:

```
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

You'd create the empty repo on GitHub's website first (green **New**
button on your GitHub homepage) before running the last two lines.

## Adding the README screenshots

Claude already prepared the 8 screenshot files for you (from the ones you
uploaded), correctly named to match the README exactly. You just need to
get them into the repo's `images/` folder:

1. Download all 8 files from the chat if you haven't already:
   `chat-1.png`, `chat-2-plus-menu.png`, `tts-studio-1-voices.png`,
   `tts-studio-2-controls.png`, `tts-studio-3-result.png`,
   `video-studio-1-edit.png`, `video-studio-2-compose.png`,
   `clip-finder-1-viral-moments.png`
2. In your browser, go directly to:
   `https://github.com/<your-username>/<your-repo-name>/upload/main/images`
   (this is your repo's normal "Upload files" page, with `/images` added
   to the end — GitHub creates that folder automatically when you upload
   into it this way, even though it doesn't exist yet).
3. Drag all 8 images into that page at once.
4. Scroll down, type a commit message like `Add screenshots`, click
   **Commit changes**.
5. Go back to your repo's main page and refresh — the README should now
   show all the screenshots.
