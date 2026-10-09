# Plateway community library

Printable track pieces, train models and furniture presets contributed by Plateway members. Plateway's app loads the
merged items into its palette (pieces in the Community group, furniture in the Furniture group) and its Test drive
train picker, each credited to its contributor.

## How items arrive

Members upload in Plateway, not here, so contributors need no GitHub account. Plateway checks each upload, and Alex
looks at it there first; anything he rejects never reaches this repository. An approved upload becomes one pull request
from the Plateway library bot, on a `submissions/<id>` branch, crediting the contributor by their Plateway display name.

- **Merge** to publish: within five minutes Plateway copies the item to its public library and it appears in the app.
- **Close** to reject: comment why first, then close. The contributor sees "Rejected" in Plateway with your last
  comment as the reason.
- A merged item that is later reverted is retired: it leaves the palette, but layouts that use it keep it.
- Takedowns happen in Plateway's admin page, which deletes the item from the library and the bot deletes its folder
  here.

## Layout

```
pieces/<id>/<version>/piece.stl        binary STL in mm, +z up, flat underside at z = 0
trains/<id>/<version>/car.glb          one car's glTF binary in Plateway's car frame
furniture/<id>/<version>/furniture.json   a Plateway FurnitureType
<kind folder>/<id>/<version>/metadata.json
```

Ids are `community_` then lowercase letters, digits and `_`. A piece prints as uploaded, so it must fit the A1 mini
as Plateway arranges it: at most 166 × 166 mm in plan (its 180 mm bed less a 7 mm margin each side) and 180 mm high.
`metadata.json` holds the name and, for a piece, its two marked ends; for a train, the set facts and car measurements
and its licence. An item adapted from another CC BY model also holds that model's source and authors under `source`.
The Plateway README ("Community library") describes the formats.

## The check

`check.yml` runs on every pull request, read-only and without secrets, and is required before merging. It runs
`check_items.py` from the base branch on each changed item folder, which runs `check.py`: the same checks Plateway
runs before the quick look. `check.py` is vendored from Plateway's `cad/library/check.py`; copy it over when that
changes. To run it locally:

```sh
python -m pip install -r requirements.txt
python check_items.py --root . pieces/community_example/1
```

## Licence

See `LICENSE`. Each item is shared under CC BY 4.0, or for a train model adapted from another CC BY model, that
model's CC BY licence, as its `metadata.json` says. An adapted piece stays CC BY 4.0 and credits its source's authors.
