# Arch User Repository — sto-warp

This directory holds the PKGBUILD that ships sto-warp to AUR users.

## Publishing flow

1. Bump `__version__` in `warp/__init__.py`.
2. Build & upload to PyPI:

   ```sh
   cd /path/to/sto-warp
   rm -rf dist build
   python -m build
   python -m twine upload dist/*
   ```

3. Update `packaging/aur/PKGBUILD`:

   - `pkgver=X.Y.Z`
   - `pkgrel=1` (bump if the PKGBUILD itself changes without a new upstream)
   - `sha256sums=('<sha of the sdist from PyPI>')`

   Quick way to grab the hash:

   ```sh
   curl -sL https://files.pythonhosted.org/packages/source/s/sto-warp/sto-warp-X.Y.Z.tar.gz \
     | sha256sum
   ```

4. Regenerate metadata and push to AUR:

   ```sh
   cd packaging/aur
   makepkg --printsrcinfo > .SRCINFO
   # First time only:
   #   git clone ssh://aur@aur.archlinux.org/sto-warp.git aur-repo
   #   cp PKGBUILD .SRCINFO aur-repo/
   git -C aur-repo add PKGBUILD .SRCINFO
   git -C aur-repo commit -m "release X.Y.Z"
   git -C aur-repo push
   ```

## Why no full dependency enumeration?

EasyOCR, scikit-image, shapely, pyclipper, python-bidi are not always
in the official Arch repos. We list them as `optdepends=` so the
install completes even when those packages are AUR-only — the user
gets a runtime warning if they try a feature that needs them.

For the fully-loaded experience the recommended path remains:

```sh
pipx install sto-warp
```

The AUR package is for users who insist on a system-managed install.
