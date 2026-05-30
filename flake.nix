{
  description =
    "CNL Final (Group 6) — Phase-2 system tools: Mininet + OVS + iperf.";
  # Just a package set (a dev environment), nothing more. Python deps stay on
  # uv; this only provides the Linux-only data-plane tools uv cannot install.
  # Orchestration (OVS bring-up, the testbed run) lives in scripts/ + Makefile.

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system:
      let pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            mininet
            openvswitch
            iperf2 # provides `iperf2` (UDP loss reporting); NOT the iperf3 alias
            iperf3
            iproute2 # tc / netem link shaping
            ethtool
            socat
            which
          ];

          # The only non-package line: expose the Nix-built (pure-Python) Mininet
          # module so the uv venv interpreter can `import mininet`. Mininet has no
          # working PyPI wheel, so uv alone cannot provide it.
          shellHook = ''
            export PYTHONPATH="${pkgs.mininet.py}/lib/${pkgs.python3.libPrefix}/site-packages''${PYTHONPATH:+:$PYTHONPATH}"
          '';
        };
      });
}
