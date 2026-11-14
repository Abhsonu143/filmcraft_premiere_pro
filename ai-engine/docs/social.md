# Social Posting (AI ▸ Social Posting…)

Post a finished video, with the titles, captions, hashtags and thumbnail from **Thumbnails, Titles &
Hashtags**, to YouTube, Instagram, Facebook and Threads: to one or several at once, now or at a set
time. Every post and how it went on each platform is listed under **Posts** (with links, errors and
Retry Failed).

## What each platform allows (checked November 2026)

| Platform | Posts from EI | Limits to know |
|---|---|---|
| YouTube video / Short | upload, title, description, tags, thumbnail, public/unlisted/private, publish time | Uploads from a personal Google Cloud project created after 28 July 2020 **stay private until the project passes YouTube's API compliance audit** ([Google docs](https://developers.google.com/youtube/v3/docs/videos/insert)). The default quota (10,000 units a day; an upload costs 1,600) allows about 6 uploads a day. A custom thumbnail needs a verified channel (youtube.com/verify). |
| Instagram Reel | upload from this Mac (resumable upload), caption, shown in the feed too | Needs an Instagram **professional** account linked to a **Facebook Page**, and a Meta app with **Facebook Login for Business** ([Meta docs](https://developers.facebook.com/documentation/instagram-platform/content-publishing/resumable-uploads)). The API cannot schedule: EI posts at the set time (EI must be open). The cover is a frame of the video. At most **5 hashtags** per post (Instagram's limit since December 2025). |
| Facebook Page video / Reel | upload from this Mac, title, description, thumbnail (video), publish time | Pages only, not personal profiles. Facebook schedules it itself (at least 10 minutes ahead; sooner is posted right away). |
| Threads | a text post (500 characters) with the link to the video just posted | Threads only takes images and videos from **public web links** ([Threads docs](https://developers.facebook.com/docs/threads/posts/)), so EI posts the text and the link. |

## One-time setup

Logins are kept in the macOS Keychain (Keychain Access ▸ "EI Social"), never in files. Disconnect
removes them.

### YouTube

1. Go to console.cloud.google.com, create a project (e.g. "EI").
2. APIs & Services ▸ Library ▸ **YouTube Data API v3** ▸ Enable.
3. APIs & Services ▸ OAuth consent screen: External; add your Google account under **Test users**.
4. APIs & Services ▸ Credentials ▸ Create credentials ▸ **OAuth client ID** ▸ Application type
   **Desktop app** ▸ Create ▸ **Download JSON**.
5. In EI: Social Posting ▸ Accounts ▸ YouTube ▸ **Connect…**, choose that JSON file. The browser opens:
   sign in with the channel's account and allow it. The Accounts tab shows the channel's name.
6. To publish publicly from EI, apply for the **YouTube API compliance audit** (the "YouTube API
   Services – Audit and Quota Extension Form"). Until it passes, uploads stay private: you can still
   use EI to upload, set the thumbnail and fill everything in, then publish in YouTube Studio if
   YouTube allows it for your project.

### Instagram and Facebook

1. Instagram app ▸ Settings ▸ Account type ▸ switch to a **Professional** (Creator or Business)
   account, and link it to a **Facebook Page** (Instagram ▸ Settings ▸ Accounts Center, or the Page's
   settings ▸ Linked accounts).
2. developers.facebook.com ▸ My Apps ▸ **Create app** ▸ use case "Other" ▸ type **Business** ▸ add the
   products **Facebook Login for Business** and **Instagram** (API setup with Facebook login). Your own
   account is the app's admin, so it works in development mode for your accounts.
3. App settings ▸ Basic: copy the **App ID** and **App secret**.
4. Tools ▸ **Graph API Explorer** ▸ choose your app ▸ Get User Access Token with the permissions
   `pages_show_list`, `pages_read_engagement`, `pages_manage_posts`, `instagram_basic`,
   `instagram_content_publish`, `business_management` ▸ Generate ▸ copy the token.
5. In EI: Social Posting ▸ Accounts ▸ Instagram & Facebook: paste the App ID, App secret and token ▸
   **Connect**. EI turns the token into a 60-day one (renewed by itself while it is still valid), finds
   your Page and its Instagram account, and shows `@your_instagram · Page "…"`.

### Threads

1. In the same Meta app (or a new one): Add use case ▸ **Access the Threads API** ▸ permissions
   `threads_basic`, `threads_content_publish`; add your Threads account as a Threads tester and accept
   the invite in Threads ▸ Settings ▸ Account ▸ Website permissions.
2. Use case ▸ Threads API ▸ generate an access token (User Token Generator) and copy it; copy the
   Threads **App secret** (App settings ▸ Basic, under Threads).
3. In EI: Social Posting ▸ Accounts ▸ Threads: paste the token (and the app secret, which makes it a
   60-day token that renews itself) ▸ **Connect**.

## Posting

1. Make the package first (AI ▸ Thumbnails, Titles & Hashtags…) from the video's clips.
2. Social Posting ▸ New Post: choose the **video** and the **package** folder (found automatically when
   it is next to the video), the **language**, then **Fill In the Words**: each platform gets its own
   title, description or caption, hashtags and thumbnail from the package. Edit anything.
3. Tick where it goes. Leave **When** empty to post now, or give a time like `2026-11-20 18:00`
   (this Mac's time). YouTube and Facebook get the post right away and publish it at that time
   themselves; Instagram and Threads are posted by EI at that time (keep EI open; EI checks every
   minute).
4. **Dry Run** goes through everything without posting (AI Jobs shows what would be posted).
   **Post Now** / **Schedule** starts it; AI ▸ AI Jobs shows the upload progress, and the Posts tab
   the result per platform with its link. **Retry Failed** posts only the failed platforms again;
   **Cancel** stops scheduled ones.

From the command line: `ei_engine.py social accounts | connect … | post --video V --targets youtube
reels threads --package P [--at "2026-11-20 18:00"] [--dry-run] | run ID [--failed] | run-due | list`.

The posts and their results are in `~/Library/Application Support/EI/social/posts.json`.
