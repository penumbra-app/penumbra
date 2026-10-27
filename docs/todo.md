# Project to-dos

## Content data: Letterboxd API access

- [x] Check the official access process and project eligibility (September 28, 2026).
- [ ] Obtain Letterboxd API access for Flick — **blocked by current eligibility rules**.
- [ ] Integrate Letterboxd data if access is granted for this project's intended use.

Letterboxd's [official API access page](https://letterboxd.com/api-beta/) says
access is request-only and applications should describe the intended use, with
the project title in the email subject. It currently excludes recommendation
projects, among other categories. Flick is a movie recommendation project, so
it does not meet the published access criteria. No application has been sent,
credentials obtained, or Letterboxd integration implemented.

Next step: revisit access if the published criteria change or Letterboxd explicitly
authorizes this use. Continue using the existing IMDb/TMDB metadata importers.
For a user's own Letterboxd history, a separate possible task is to support
user-provided account exports; the access page points to its export facilities.
An export importer has not been implemented.

## Content scoring: Netflix watch-history import

- [ ] Implement import of a user-provided Netflix viewing-history export.
- [ ] Match watched titles to the movie catalog, handling ambiguous titles,
  unmatched entries, duplicates, and TV episodes explicitly.
- [ ] Use matched viewing history as an implicit preference signal that can
  adjust recommendation scores. Treat watching as weaker evidence than an
  explicit rating; a watched movie is not automatically a liked movie.
- [ ] Add a configurable weight and cap for the history-based score adjustment,
  with supported reason signals and a neutral fallback when history is absent.
- [ ] Evaluate with and without this signal on the same split, using only viewing
  events available before each prediction. Retain it if validation supports it.

Status: planned; no Netflix history importer or scoring adjustment is implemented.
