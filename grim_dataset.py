import json
import csv
import os
import re
import warnings
import numpy as np

C0 = 299_792_458.0


class RcsGrid:
    """Container for gridded RCS data with axis metadata and helpers."""

    def __init__(
        self,
        azimuths,
        elevations,
        frequencies,
        polarizations,
        rcs=None,
        rcs_power=None,
        rcs_phase=None,
        rcs_domain: str | None = None,
        source_path: str | None = None,
        history: str | None = None,
        units: dict | None = None,
    ):
        """Build a grid from axis arrays and power/phase-backed RCS samples.

        Use when loading data from files or constructing an in-memory grid.

        Args:
            azimuths: 1D sequence of azimuth values (deg).
            elevations: 1D sequence of elevation values (deg).
            frequencies: 1D sequence of frequency values (GHz or Hz).
            polarizations: 1D sequence of polarization labels.
            rcs: Optional complex field samples shaped (az, el, f, pol).
            rcs_power: Optional linear-power samples shaped (az, el, f, pol).
            rcs_phase: Optional phase samples (radians) shaped (az, el, f, pol).
                Use NaN where phase is unknown.
            rcs_domain: Optional domain tag metadata.
            source_path: Optional source path for provenance.
            history: Optional history string.
            units: Optional units dict (e.g., {"azimuth": "deg", "frequency": "GHz"}).

        Raises:
            ValueError: if shapes do not match the expected grid.
        """

        self.azimuths = np.asarray(azimuths)
        self.elevations = np.asarray(elevations)
        self.frequencies = np.asarray(frequencies)
        self.polarizations = np.asarray(polarizations)

        expected = (len(self.azimuths), len(self.elevations), len(self.frequencies), len(self.polarizations))

        complex_arr = None
        if rcs is not None:
            rcs_arr = np.asarray(rcs)
            if rcs_arr.shape == expected + (2,):
                complex_arr = np.asarray(rcs_arr[..., 0] + 1j * rcs_arr[..., 1], dtype=np.complex64)
            elif rcs_arr.shape == expected:
                if np.iscomplexobj(rcs_arr):
                    complex_arr = np.asarray(rcs_arr, dtype=np.complex64)
                elif rcs_power is None:
                    # Real-valued rcs input is treated as linear power when explicit power is not provided.
                    rcs_power = np.asarray(rcs_arr, dtype=np.float32)
            else:
                raise ValueError(f"rcs shape {rcs_arr.shape} != {expected}")

        if rcs_power is not None:
            power_arr = np.asarray(rcs_power, dtype=np.float32)
            if power_arr.shape != expected:
                raise ValueError(f"rcs_power shape {power_arr.shape} != {expected}")
        elif complex_arr is not None:
            power_arr = np.abs(complex_arr) ** 2
        else:
            raise ValueError("provide complex rcs samples and/or rcs_power")

        if rcs_phase is not None:
            phase_arr = np.asarray(rcs_phase, dtype=np.float32)
            if phase_arr.shape != expected:
                raise ValueError(f"rcs_phase shape {phase_arr.shape} != {expected}")
        elif complex_arr is not None:
            phase_arr = np.angle(complex_arr).astype(np.float32)
        else:
            phase_arr = np.full(expected, np.nan, dtype=np.float32)

        power_clean = self._clean_power(power_arr)
        phase_clean = self._clean_phase(phase_arr)
        phase_clean[~np.isfinite(power_clean)] = np.nan

        self.rcs_power = power_clean
        self.rcs_phase = phase_clean
        domain = str(rcs_domain or "").strip().lower()
        if domain not in {"complex_amplitude", "linear_rcs", "power_phase"}:
            domain = "power_phase"
        self.rcs_domain = domain
        self.power_domain = "linear_rcs"
        self.source_path = source_path
        self.history = history
        self.units = units or {}

    @staticmethod
    def _clean_power(power_value):
        power = np.asarray(power_value, dtype=np.float32)
        finite = np.isfinite(power)
        out = np.full(power.shape, np.nan, dtype=np.float32)
        out[finite] = np.maximum(power[finite], 0.0)
        return out

    @staticmethod
    def _clean_phase(phase_value):
        phase = np.array(phase_value, dtype=np.float32, copy=True)
        phase[~np.isfinite(phase)] = np.nan
        return phase

    @staticmethod
    def _complex_from_power_phase(power_value, phase_value):
        power = np.asarray(power_value, dtype=np.float32)
        phase = np.asarray(phase_value, dtype=np.float32)
        if power.shape != phase.shape:
            raise ValueError(f"power/phase shapes {power.shape}/{phase.shape} do not match")
        out = np.full(power.shape, np.nan + 1j * np.nan, dtype=np.complex64)
        valid = np.isfinite(power) & np.isfinite(phase)
        if np.any(valid):
            out[valid] = (np.sqrt(power[valid]) * np.exp(1j * phase[valid])).astype(np.complex64)
        return out

    @property
    def rcs(self):
        """Complex RCS values derived from stored linear power and phase."""
        return self._complex_from_power_phase(self.rcs_power, self.rcs_phase)

    def __len__(self):
        """Return total number of complex samples in the grid."""
        return self.rcs_power.size

    def get(self, az_idx, el_idx, f_idx, p_idx):
        """Fetch a single sample by axis indices.

        Args:
            az_idx: Azimuth index.
            el_idx: Elevation index.
            f_idx: Frequency index.
            p_idx: Polarization index.

        Returns:
            dict with axis values and complex RCS sample.
        """
        return {
            "azimuth": self.azimuths[az_idx],
            "elevation": self.elevations[el_idx],
            "frequency": self.frequencies[f_idx],
            "polarization": self.polarizations[p_idx],
            "rcs": self.rcs[az_idx, el_idx, f_idx, p_idx],
        }

    def get_axis(self, name):
        """Return a single axis array by name.

        Use when you need a specific axis without unpacking all axes.

        Args:
            name: One of "azimuth", "elevation", "frequency", "polarization".

        Returns:
            Numpy array for the requested axis.
        """
        if name == "azimuth":
            return self.azimuths
        if name == "elevation":
            return self.elevations
        if name == "frequency":
            return self.frequencies
        if name == "polarization":
            return self.polarizations
        raise ValueError(f"unknown axis name: {name}")

    def get_axes(self):
        """Return all axis arrays in a dict."""
        return {
            "azimuths": self.azimuths,
            "elevations": self.elevations,
            "frequencies": self.frequencies,
            "polarizations": self.polarizations,
        }

    def _assert_compatible(self, other):
        """Validate another grid for element-wise operations.

        Use before coherent/incoherent add/subtract operations.

        Args:
            other: Another RcsGrid instance.

        Raises:
            TypeError: if other is not an RcsGrid.
            ValueError: if axes or shapes differ.
        """
        if not isinstance(other, RcsGrid):
            raise TypeError("other must be an RcsGrid")
        if self.rcs_power.shape != other.rcs_power.shape:
            raise ValueError(f"rcs shape {other.rcs_power.shape} != {self.rcs_power.shape}")
        if not np.array_equal(self.azimuths, other.azimuths):
            raise ValueError("azimuth axis mismatch")
        if not np.array_equal(self.elevations, other.elevations):
            raise ValueError("elevation axis mismatch")
        if not np.array_equal(self.frequencies, other.frequencies):
            raise ValueError("frequency axis mismatch")
        if not np.array_equal(self.polarizations, other.polarizations):
            raise ValueError("polarization axis mismatch")

    def coherent_add(self, other):
        """Coherently add two grids (complex sum).

        Use when phases are aligned and you want field-level addition.

        Args:
            other: Another RcsGrid with identical axes.

        Returns:
            New RcsGrid with rcs = self.rcs + other.rcs.
        """
        self._assert_compatible(other)
        rcs_out = self.rcs + other.rcs
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_out,
            rcs_domain="power_phase",
        )

    def coherent_add_many(self, *grids):
        """Coherently add multiple grids (complex sum).

        Use when phases are aligned and you want field-level addition.

        Args:
            *grids: One or more RcsGrid instances.

        Returns:
            New RcsGrid with rcs = self.rcs + sum(grid.rcs).
        """
        if not grids:
            return self
        total = np.array(self.rcs, copy=True)
        for grid in grids:
            self._assert_compatible(grid)
            total = total + grid.rcs
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            total,
            rcs_domain="power_phase",
        )

    def coherent_subtract(self, other):
        """Coherently subtract two grids (complex difference).

        Use when phases are aligned and you want field-level subtraction.

        Args:
            other: Another RcsGrid with identical axes.

        Returns:
            New RcsGrid with rcs = self.rcs - other.rcs.
        """
        self._assert_compatible(other)
        rcs_out = self.rcs - other.rcs
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_out,
            rcs_domain="power_phase",
        )

    def incoherent_add(self, other):
        """Incoherently add two grids (magnitude sum).

        Use when phases are unrelated and you want power-level addition.

        Args:
            other: Another RcsGrid with identical axes.

        Returns:
            New RcsGrid with linear power = self.rcs_power + other.rcs_power.
        """
        self._assert_compatible(other)
        power_sum = self.rcs_power + other.rcs_power
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_power=power_sum,
            rcs_phase=np.full(power_sum.shape, np.nan, dtype=np.float32),
            rcs_domain="power_phase",
        )

    def incoherent_add_many(self, *grids):
        """Incoherently add multiple grids (magnitude sum).

        Use when phases are unrelated and you want power-level addition.

        Args:
            *grids: One or more RcsGrid instances.

        Returns:
            New RcsGrid with linear power = self.rcs_power + sum(grid.rcs_power).
        """
        if not grids:
            return self
        total = np.array(self.rcs_power, copy=True)
        for grid in grids:
            self._assert_compatible(grid)
            total = total + grid.rcs_power
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_power=total,
            rcs_phase=np.full(total.shape, np.nan, dtype=np.float32),
            rcs_domain="power_phase",
        )

    def incoherent_subtract(self, other):
        """Incoherently subtract two grids (magnitude difference).

        Use when phases are unrelated and you want power-level subtraction.

        Args:
            other: Another RcsGrid with identical axes.

        Returns:
            New RcsGrid with linear power = max(self.rcs_power - other.rcs_power, 0).
        """
        self._assert_compatible(other)
        power_diff = self.rcs_power - other.rcs_power
        power_diff = np.maximum(power_diff, 0.0)
        return self._new_grid(
            self.azimuths,
            self.elevations,
            self.frequencies,
            self.polarizations,
            rcs_power=power_diff,
            rcs_phase=np.full(power_diff.shape, np.nan, dtype=np.float32),
            rcs_domain="power_phase",
        )

    def align_to(self, other, mode="exact"):
        """Align this grid to another grid's axes.

        Modes:
            exact: require identical axes (returns self on success).
            intersect: keep only axis values present in both grids.
            interp: interpolate numeric axes to match other (no extrapolation).

        Args:
            other: Another RcsGrid instance.
            mode: "exact", "intersect", or "interp".

        Returns:
            New RcsGrid aligned to other's axes.
        """
        if not isinstance(other, RcsGrid):
            raise TypeError("other must be an RcsGrid")

        if mode == "exact":
            self._assert_compatible(other)
            return self
        if mode not in ("intersect", "interp"):
            raise ValueError("mode must be 'exact', 'intersect', or 'interp'")

        if mode == "intersect":
            def _match_axis(axis_self, axis_other, tol=1e-6):
                axis_self = np.asarray(axis_self)
                axis_other = np.asarray(axis_other)
                is_numeric = np.issubdtype(axis_self.dtype, np.number)
                keep_other = []
                indices_self = []
                for value in axis_other:
                    if is_numeric:
                        matches = np.where(np.isclose(axis_self, value, atol=tol, rtol=0.0))[0]
                    else:
                        matches = np.where(axis_self == value)[0]
                    if matches.size > 0:
                        keep_other.append(value)
                        indices_self.append(int(matches[0]))
                if not indices_self:
                    raise ValueError("no overlapping axis values for intersect")
                return np.asarray(keep_other), indices_self

            az_new, az_idx = _match_axis(self.azimuths, other.azimuths)
            el_new, el_idx = _match_axis(self.elevations, other.elevations)
            f_new, f_idx = _match_axis(self.frequencies, other.frequencies)
            pol_new, pol_idx = _match_axis(self.polarizations, other.polarizations, tol=0.0)
            pwr_new = self.rcs_power[np.ix_(az_idx, el_idx, f_idx, pol_idx)]
            phs_new = self.rcs_phase[np.ix_(az_idx, el_idx, f_idx, pol_idx)]
            return self._new_grid(
                az_new,
                el_new,
                f_new,
                pol_new,
                rcs_power=pwr_new,
                rcs_phase=phs_new,
                rcs_domain="power_phase",
            )

        # interp mode
        if not np.array_equal(self.polarizations, other.polarizations):
            raise ValueError("polarization axis mismatch for interp")

        def _check_sorted(axis, name):
            axis = np.asarray(axis)
            if axis.size < 2:
                return
            if not np.all(np.diff(axis) > 0):
                raise ValueError(f"{name} axis must be strictly increasing for interp")

        _check_sorted(self.azimuths, "azimuth")
        _check_sorted(self.elevations, "elevation")
        _check_sorted(self.frequencies, "frequency")
        _check_sorted(other.azimuths, "azimuth")
        _check_sorted(other.elevations, "elevation")
        _check_sorted(other.frequencies, "frequency")

        def _interp_complex_axis(data, x_old, x_new, axis):
            x_old = np.asarray(x_old, dtype=float)
            x_new = np.asarray(x_new, dtype=float)
            if x_new.min() < x_old.min() or x_new.max() > x_old.max():
                raise ValueError("interp would require extrapolation")
            moved = np.moveaxis(data, axis, 0)
            flat = moved.reshape(moved.shape[0], -1)
            real = np.empty((x_new.size, flat.shape[1]), dtype=float)
            imag = np.empty((x_new.size, flat.shape[1]), dtype=float)
            for i in range(flat.shape[1]):
                real[:, i] = np.interp(x_new, x_old, flat[:, i].real)
                imag[:, i] = np.interp(x_new, x_old, flat[:, i].imag)
            combined = real + 1j * imag
            out = combined.reshape((x_new.size,) + moved.shape[1:])
            return np.moveaxis(out, 0, axis)

        def _interp_real_axis(data, x_old, x_new, axis):
            x_old = np.asarray(x_old, dtype=float)
            x_new = np.asarray(x_new, dtype=float)
            if x_new.min() < x_old.min() or x_new.max() > x_old.max():
                raise ValueError("interp would require extrapolation")
            moved = np.moveaxis(data, axis, 0)
            flat = moved.reshape(moved.shape[0], -1)
            out_flat = np.empty((x_new.size, flat.shape[1]), dtype=np.float32)
            for i in range(flat.shape[1]):
                out_flat[:, i] = np.interp(x_new, x_old, flat[:, i]).astype(np.float32)
            out = out_flat.reshape((x_new.size,) + moved.shape[1:])
            return np.moveaxis(out, 0, axis)

        phase_missing = np.isfinite(self.rcs_power) & ~np.isfinite(self.rcs_phase)
        if np.any(phase_missing):
            power_interp = _interp_real_axis(self.rcs_power, self.azimuths, other.azimuths, axis=0)
            power_interp = _interp_real_axis(power_interp, self.elevations, other.elevations, axis=1)
            power_interp = _interp_real_axis(power_interp, self.frequencies, other.frequencies, axis=2)
            phase_interp = np.full(power_interp.shape, np.nan, dtype=np.float32)
            return self._new_grid(
                other.azimuths,
                other.elevations,
                other.frequencies,
                other.polarizations,
                rcs_power=power_interp,
                rcs_phase=phase_interp,
                rcs_domain="power_phase",
            )

        rcs_interp = _interp_complex_axis(self.rcs, self.azimuths, other.azimuths, axis=0)
        rcs_interp = _interp_complex_axis(rcs_interp, self.elevations, other.elevations, axis=1)
        rcs_interp = _interp_complex_axis(rcs_interp, self.frequencies, other.frequencies, axis=2)
        return self._new_grid(
            other.azimuths,
            other.elevations,
            other.frequencies,
            other.polarizations,
            rcs_interp,
            rcs_domain="power_phase",
        )

    @staticmethod
    def _as_list(value):
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            return [value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple, set)):
            return list(value)
        return [value]

    @staticmethod
    def _axis_value_match(axis_arr, value, tol=1e-6):
        axis_arr = np.asarray(axis_arr)
        if np.issubdtype(axis_arr.dtype, np.number) and isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            return np.where(np.isclose(axis_arr, float(value), atol=tol, rtol=0.0))[0]
        return np.where(axis_arr == value)[0]

    @staticmethod
    def _indices_for_axis_values(axis_arr, values, tol=1e-6):
        axis_arr = np.asarray(axis_arr)
        indices = []
        for value in values:
            matches = RcsGrid._axis_value_match(axis_arr, value, tol=tol)
            if matches.size == 0:
                return None
            idx = int(matches[0])
            if idx not in indices:
                indices.append(idx)
        return indices

    @staticmethod
    def _axis_union(axis_arrays, tol=1e-6):
        if not axis_arrays:
            return np.asarray([])
        union_values = []
        first_dtype = np.asarray(axis_arrays[0]).dtype
        numeric_axis = np.issubdtype(first_dtype, np.number)
        for axis_arr in axis_arrays:
            for value in np.asarray(axis_arr):
                plain = value.item() if isinstance(value, np.generic) else value
                if numeric_axis:
                    exists = any(
                        np.isclose(float(existing), float(plain), atol=tol, rtol=0.0)
                        for existing in union_values
                    )
                else:
                    exists = any(existing == plain for existing in union_values)
                if not exists:
                    union_values.append(plain)
        if numeric_axis:
            union_values.sort(key=float)
        return np.asarray(union_values)

    @staticmethod
    def _axis_intersection(axis_arrays, tol=1e-6):
        if not axis_arrays:
            return np.asarray([])
        common = [
            value.item() if isinstance(value, np.generic) else value
            for value in np.asarray(axis_arrays[0])
        ]
        for axis_arr in axis_arrays[1:]:
            common = [
                value
                for value in common
                if RcsGrid._axis_value_match(axis_arr, value, tol=tol).size > 0
            ]
            if not common:
                break
        return np.asarray(common)

    @classmethod
    def _ensure_grids(cls, grids):
        checked = []
        for grid in grids:
            if not isinstance(grid, cls):
                raise TypeError("all inputs must be RcsGrid instances")
            checked.append(grid)
        if not checked:
            raise ValueError("at least one grid is required")
        return checked

    def _new_grid(
        self,
        azimuths,
        elevations,
        frequencies,
        polarizations,
        rcs=None,
        *,
        rcs_power=None,
        rcs_phase=None,
        rcs_domain=None,
        history=None,
    ):
        return RcsGrid(
            azimuths,
            elevations,
            frequencies,
            polarizations,
            rcs,
            rcs_power=rcs_power,
            rcs_phase=rcs_phase,
            rcs_domain=(self.rcs_domain if rcs_domain is None else rcs_domain),
            source_path=self.source_path,
            history=history if history is not None else self.history,
            units=dict(self.units),
        )

    def _power_from_values(self, rcs_value):
        values_raw = np.asarray(rcs_value)
        if np.iscomplexobj(values_raw):
            values = np.asarray(values_raw, dtype=np.complex128)
            power = np.abs(values) ** 2
        else:
            power = np.asarray(values_raw, dtype=float)
        power = np.asarray(power, dtype=float)
        finite = np.isfinite(power)
        out = np.zeros_like(power, dtype=float)
        out[finite] = np.maximum(power[finite], 0.0)
        out[~finite] = np.nan
        return out

    def _amplitude_from_power(self, power_value):
        power = self._clean_power(power_value)
        zero_phase = np.zeros(power.shape, dtype=np.float32)
        return self._complex_from_power_phase(power, zero_phase)

    def rcs_to_linear(self, rcs_value):
        """Convert complex field or real-power values to linear power."""
        return self._power_from_values(rcs_value)

    def linear_to_dbsm(self, linear_value, eps=1e-12):
        linear = np.asarray(linear_value, dtype=float)
        linear = np.where(np.isfinite(linear), linear, np.nan)
        linear = np.maximum(linear, eps)
        return 10.0 * np.log10(linear)

    def _frequency_value_to_hz(self, frequency_value):
        freq = np.asarray(frequency_value, dtype=float)
        unit = str((self.units or {}).get("frequency", "GHz")).strip().lower()
        if unit == "hz":
            return freq
        if unit == "mhz":
            return freq * 1.0e6
        if unit == "khz":
            return freq * 1.0e3
        return freq * 1.0e9

    def linear_to_dbke(self, linear_value, frequency_value, eps=1e-12):
        linear = np.asarray(linear_value, dtype=float)
        linear = np.where(np.isfinite(linear), linear, np.nan)
        linear = np.maximum(linear, eps)
        freq_hz = self._frequency_value_to_hz(frequency_value)
        freq_hz = np.asarray(freq_hz, dtype=float)
        freq_hz = np.where(np.isfinite(freq_hz) & (freq_hz > 0.0), freq_hz, np.nan)
        factor = (2.0 * np.pi * freq_hz) / C0
        return 10.0 * np.log10(factor * linear)

    def dbke_to_linear(self, dbke_value, frequency_value):
        dbke = np.asarray(dbke_value, dtype=float)
        freq_hz = self._frequency_value_to_hz(frequency_value)
        freq_hz = np.asarray(freq_hz, dtype=float)
        factor = np.where(np.isfinite(freq_hz) & (freq_hz > 0.0), C0 / (2.0 * np.pi * freq_hz), np.nan)
        return factor * (10.0 ** (dbke / 10.0))

    def default_log_unit(self):
        raw = str((self.units or {}).get("rcs_log_unit", "dBsm")).strip().lower()
        return "dBke" if raw == "dbke" else "dBsm"

    def linear_to_default_db(self, linear_value, frequency_value=None, eps=1e-12):
        if self.default_log_unit().lower() == "dbke":
            if frequency_value is None:
                raise ValueError("frequency_value is required for dBke conversion")
            return self.linear_to_dbke(linear_value, frequency_value, eps=eps)
        return self.linear_to_dbsm(linear_value, eps=eps)

    def axis_crop(
        self,
        *,
        azimuths=None,
        elevations=None,
        frequencies=None,
        polarizations=None,
        azimuth_range=None,
        elevation_range=None,
        frequency_range=None,
        azimuth_min=None,
        azimuth_max=None,
        elevation_min=None,
        elevation_max=None,
        frequency_min=None,
        frequency_max=None,
        tol=1e-6,
    ):
        """Return a grid cropped by explicit axis values and/or numeric ranges."""

        def _resolve_range(raw_range, vmin, vmax):
            if raw_range is not None:
                if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
                    raise ValueError("axis range must be a 2-item [min, max] sequence")
                return raw_range[0], raw_range[1]
            if vmin is None and vmax is None:
                return None
            return vmin, vmax

        azimuth_range = _resolve_range(azimuth_range, azimuth_min, azimuth_max)
        elevation_range = _resolve_range(elevation_range, elevation_min, elevation_max)
        frequency_range = _resolve_range(frequency_range, frequency_min, frequency_max)

        def _axis_indices(axis_arr, axis_values, axis_range, axis_name, axis_tol):
            all_indices = list(range(len(axis_arr)))
            values = self._as_list(axis_values)
            if values is not None:
                selected = self._indices_for_axis_values(axis_arr, values, tol=axis_tol)
                if selected is None:
                    raise ValueError(f"{axis_name} contains value(s) not present in dataset")
                indices = selected
            else:
                indices = all_indices

            if axis_range is not None:
                lo, hi = axis_range
                if lo is not None:
                    lo = float(lo)
                if hi is not None:
                    hi = float(hi)
                if lo is not None and hi is not None and lo > hi:
                    lo, hi = hi, lo

                axis_num = np.asarray(axis_arr, dtype=float)
                range_mask = np.ones(axis_num.shape[0], dtype=bool)
                if lo is not None:
                    range_mask &= axis_num >= (lo - axis_tol)
                if hi is not None:
                    range_mask &= axis_num <= (hi + axis_tol)
                range_idx = set(np.where(range_mask)[0].tolist())
                indices = [idx for idx in indices if idx in range_idx]

            if not indices:
                raise ValueError(f"{axis_name} crop produced no samples")
            return indices

        az_idx = _axis_indices(self.azimuths, azimuths, azimuth_range, "azimuth", tol)
        el_idx = _axis_indices(self.elevations, elevations, elevation_range, "elevation", tol)
        f_idx = _axis_indices(self.frequencies, frequencies, frequency_range, "frequency", tol)
        p_idx = _axis_indices(self.polarizations, polarizations, None, "polarization", 0.0)

        return self._new_grid(
            self.azimuths[az_idx],
            self.elevations[el_idx],
            self.frequencies[f_idx],
            self.polarizations[p_idx],
            rcs_power=self.rcs_power[np.ix_(az_idx, el_idx, f_idx, p_idx)],
            rcs_phase=self.rcs_phase[np.ix_(az_idx, el_idx, f_idx, p_idx)],
        )

    def mirror_about_azimuth(self, azimuth_deg: float):
        """Mirror azimuth axis about a reference angle and return a new grid.

        The transformed axis is `az' = 2*azimuth_deg - az`. Output azimuths are
        sorted ascending, with samples reordered to match.
        """
        about = float(azimuth_deg)
        if not np.isfinite(about):
            raise ValueError("mirror azimuth must be finite")

        az = np.asarray(self.azimuths, dtype=float)
        mirrored_az = (2.0 * about) - az
        order = np.argsort(mirrored_az, kind="stable")

        return self._new_grid(
            mirrored_az[order],
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=self.rcs_power[order, :, :, :],
            rcs_phase=self.rcs_phase[order, :, :, :],
            rcs_domain="power_phase",
        )

    def swap_elevation_azimuth(self):
        """Swap the elevation and azimuth axes and return a new grid."""
        return self._new_grid(
            np.array(self.elevations, copy=True),
            np.array(self.azimuths, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=np.swapaxes(self.rcs_power, 0, 1).copy(),
            rcs_phase=np.swapaxes(self.rcs_phase, 0, 1).copy(),
            rcs_domain="power_phase",
        )

    def shift_azimuth(self, delta_deg: float):
        """Shift azimuth axis by a constant offset and return a new grid."""
        delta = float(delta_deg)
        if not np.isfinite(delta):
            raise ValueError("azimuth shift must be finite")
        shifted_az = np.asarray(self.azimuths, dtype=float) + delta
        return self._new_grid(
            shifted_az,
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=np.array(self.rcs_phase, copy=True),
            rcs_domain="power_phase",
        )

    def shift_elevation(self, delta_deg: float):
        """Shift elevation axis by a constant offset and return a new grid."""
        delta = float(delta_deg)
        if not np.isfinite(delta):
            raise ValueError("elevation shift must be finite")
        shifted_el = np.asarray(self.elevations, dtype=float) + delta
        return self._new_grid(
            np.array(self.azimuths, copy=True),
            shifted_el,
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=np.array(self.rcs_power, copy=True),
            rcs_phase=np.array(self.rcs_phase, copy=True),
            rcs_domain="power_phase",
        )

    def combine_elevation_pair_to_azimuth_360(
        self,
        elevation_lo: float | None = None,
        elevation_hi: float | None = None,
        *,
        azimuth_shift_deg: float = 180.0,
        tol: float = 1e-6,
    ):
        """Stitch two elevation cuts into one 0-360 azimuth cut.

        The lower-elevation cut keeps its original azimuth values. The higher
        cut is shifted by `azimuth_shift_deg` and merged onto the same output
        elevation plane. Overlap bins keep the lower-elevation data.
        """

        el_axis = np.asarray(self.elevations, dtype=float)
        if el_axis.size < 2:
            raise ValueError("need at least 2 elevation values to combine into 360 azimuth")

        if elevation_lo is None or elevation_hi is None:
            finite = el_axis[np.isfinite(el_axis)]
            if finite.size < 2:
                raise ValueError("elevation axis has fewer than 2 finite values")
            lo_value = float(np.min(finite))
            hi_value = float(np.max(finite))
        else:
            lo_value = float(elevation_lo)
            hi_value = float(elevation_hi)

        if not np.isfinite(lo_value) or not np.isfinite(hi_value):
            raise ValueError("elevation pair values must be finite")
        if np.isclose(lo_value, hi_value, atol=tol, rtol=0.0):
            raise ValueError("elevation pair values must be distinct")

        lo_matches = self._axis_value_match(self.elevations, lo_value, tol=tol)
        hi_matches = self._axis_value_match(self.elevations, hi_value, tol=tol)
        if lo_matches.size == 0 or hi_matches.size == 0:
            raise ValueError("requested elevation pair not found in dataset")

        lo_idx = int(lo_matches[0])
        hi_idx = int(hi_matches[0])
        az_shift = float(azimuth_shift_deg)
        if not np.isfinite(az_shift):
            raise ValueError("azimuth shift must be finite")

        az_base = np.asarray(self.azimuths, dtype=float)
        if az_base.size == 0:
            raise ValueError("dataset has no azimuth samples")

        az_lo = np.array(az_base, copy=True)
        az_hi = np.array(az_base, copy=True) + az_shift
        az_merged = self._axis_union([az_lo, az_hi], tol=tol)
        if az_merged.size == 0:
            raise ValueError("combined azimuth axis is empty")

        out_shape = (len(az_merged), 1, len(self.frequencies), len(self.polarizations))
        out_power = np.full(out_shape, np.nan, dtype=np.float32)
        out_phase = np.full(out_shape, np.nan, dtype=np.float32)

        lo_target_idx = self._indices_for_axis_values(az_merged, az_lo, tol=tol)
        hi_target_idx = self._indices_for_axis_values(az_merged, az_hi, tol=tol)
        if lo_target_idx is None or hi_target_idx is None:
            raise ValueError("failed to align azimuth bins during elevation combine")

        lo_power = self.rcs_power[:, lo_idx, :, :]
        lo_phase = self.rcs_phase[:, lo_idx, :, :]
        hi_power = self.rcs_power[:, hi_idx, :, :]
        hi_phase = self.rcs_phase[:, hi_idx, :, :]

        for src_idx, dst_idx in enumerate(lo_target_idx):
            out_power[dst_idx, 0, :, :] = lo_power[src_idx, :, :]
            out_phase[dst_idx, 0, :, :] = lo_phase[src_idx, :, :]

        for src_idx, dst_idx in enumerate(hi_target_idx):
            existing_power = out_power[dst_idx, 0, :, :]
            existing_phase = out_phase[dst_idx, 0, :, :]
            incoming_power = hi_power[src_idx, :, :]
            incoming_phase = hi_phase[src_idx, :, :]

            take_power = (~np.isfinite(existing_power)) & np.isfinite(incoming_power)
            existing_power[take_power] = incoming_power[take_power]

            take_phase = np.isfinite(incoming_phase) & (
                (~np.isfinite(existing_phase)) | take_power
            )
            existing_phase[take_phase] = incoming_phase[take_phase]

        return self._new_grid(
            az_merged,
            np.asarray([el_axis[lo_idx]], dtype=float),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=out_power,
            rcs_phase=out_phase,
            rcs_domain="power_phase",
        )

    def medianize_azimuth(self, window_deg: float, slide_deg: float):
        """Sliding-window median over azimuth (in degrees).

        Window is centered at each output azimuth location. Output azimuths run
        from min(azimuth) to max(azimuth) with the requested slide step.
        """
        window = float(window_deg)
        slide = float(slide_deg)
        if not np.isfinite(window) or window <= 0.0:
            raise ValueError("medianize window must be finite and > 0")
        if not np.isfinite(slide) or slide <= 0.0:
            raise ValueError("medianize slide must be finite and > 0")

        az = np.asarray(self.azimuths, dtype=float)
        if az.size == 0:
            raise ValueError("dataset has no azimuth samples")
        if not np.all(np.isfinite(az)):
            raise ValueError("medianize requires finite azimuth values")

        order = np.argsort(az, kind="stable")
        az_sorted = az[order]
        pwr_sorted = self.rcs_power[order, :, :, :]

        start = float(az_sorted[0])
        stop = float(az_sorted[-1])
        tol = 1e-9
        count = int(np.floor((stop - start) / slide + tol)) + 1
        out_az = start + np.arange(count, dtype=float) * slide
        if out_az.size == 0:
            out_az = np.asarray([start], dtype=float)
        if out_az[-1] < stop - tol:
            out_az = np.append(out_az, stop)

        out_shape = (out_az.size,) + pwr_sorted.shape[1:]
        out_power = np.full(out_shape, np.nan, dtype=np.float32)
        out_phase = np.full(out_shape, np.nan, dtype=np.float32)
        half = 0.5 * window

        for i, center in enumerate(out_az):
            lo = center - half
            hi = center + half
            idx = np.where((az_sorted >= lo - tol) & (az_sorted <= hi + tol))[0]
            if idx.size == 0:
                continue
            segment = pwr_sorted[idx, :, :, :]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(segment, axis=0)
            out_power[i, :, :, :] = np.asarray(med, dtype=np.float32)

        return self._new_grid(
            out_az,
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
            rcs_power=out_power,
            rcs_phase=out_phase,
            rcs_domain="power_phase",
        )

    @classmethod
    def join_many(cls, *grids, tol=1e-6):
        """Join datasets on union axes; later grids overwrite overlaps."""
        grids = cls._ensure_grids(grids)
        if len(grids) == 1:
            grid = grids[0]
            return grid._new_grid(
                np.array(grid.azimuths, copy=True),
                np.array(grid.elevations, copy=True),
                np.array(grid.frequencies, copy=True),
                np.array(grid.polarizations, copy=True),
                rcs_power=np.array(grid.rcs_power, copy=True),
                rcs_phase=np.array(grid.rcs_phase, copy=True),
                rcs_domain="power_phase",
            )

        az_union = cls._axis_union([grid.azimuths for grid in grids], tol=tol)
        el_union = cls._axis_union([grid.elevations for grid in grids], tol=tol)
        f_union = cls._axis_union([grid.frequencies for grid in grids], tol=tol)
        p_union = cls._axis_union([grid.polarizations for grid in grids], tol=0.0)

        shape = (len(az_union), len(el_union), len(f_union), len(p_union))
        joined_power = np.full(shape, np.nan, dtype=np.float32)
        joined_phase = np.full(shape, np.nan, dtype=np.float32)

        for grid in grids:
            az_idx = cls._indices_for_axis_values(az_union, grid.azimuths, tol=tol)
            el_idx = cls._indices_for_axis_values(el_union, grid.elevations, tol=tol)
            f_idx = cls._indices_for_axis_values(f_union, grid.frequencies, tol=tol)
            p_idx = cls._indices_for_axis_values(p_union, grid.polarizations, tol=0.0)
            if az_idx is None or el_idx is None or f_idx is None or p_idx is None:
                raise ValueError("failed to align a dataset during join")
            joined_power[np.ix_(az_idx, el_idx, f_idx, p_idx)] = grid.rcs_power
            joined_phase[np.ix_(az_idx, el_idx, f_idx, p_idx)] = grid.rcs_phase

        last = grids[-1]
        return cls(
            az_union,
            el_union,
            f_union,
            p_union,
            rcs_power=joined_power,
            rcs_phase=joined_phase,
            rcs_domain="power_phase",
            source_path=last.source_path,
            history=last.history,
            units=dict(last.units),
        )

    @classmethod
    def overlap_many(cls, *grids, tol=1e-6):
        """Return one cropped dataset per input, all on common overlap axes."""
        grids = cls._ensure_grids(grids)
        if len(grids) == 1:
            return [grids[0]]

        az_common = cls._axis_intersection([grid.azimuths for grid in grids], tol=tol)
        el_common = cls._axis_intersection([grid.elevations for grid in grids], tol=tol)
        f_common = cls._axis_intersection([grid.frequencies for grid in grids], tol=tol)
        p_common = cls._axis_intersection([grid.polarizations for grid in grids], tol=0.0)

        if (
            az_common.size == 0
            or el_common.size == 0
            or f_common.size == 0
            or p_common.size == 0
        ):
            raise ValueError("no overlap across one or more axes")

        overlap_grids = []
        for grid in grids:
            az_idx = cls._indices_for_axis_values(grid.azimuths, az_common, tol=tol)
            el_idx = cls._indices_for_axis_values(grid.elevations, el_common, tol=tol)
            f_idx = cls._indices_for_axis_values(grid.frequencies, f_common, tol=tol)
            p_idx = cls._indices_for_axis_values(grid.polarizations, p_common, tol=0.0)
            if az_idx is None or el_idx is None or f_idx is None or p_idx is None:
                raise ValueError("failed to align a dataset during overlap")

            overlap_grids.append(
                cls(
                    az_common,
                    el_common,
                    f_common,
                    p_common,
                    rcs_power=grid.rcs_power[np.ix_(az_idx, el_idx, f_idx, p_idx)],
                    rcs_phase=grid.rcs_phase[np.ix_(az_idx, el_idx, f_idx, p_idx)],
                    rcs_domain="power_phase",
                    source_path=grid.source_path,
                    history=grid.history,
                    units=dict(grid.units),
                )
            )

        return overlap_grids

    def difference(self, other, mode="coherent"):
        """Difference between two compatible datasets."""
        if mode == "coherent":
            return self.coherent_subtract(other)
        if mode == "incoherent":
            return self.incoherent_subtract(other)
        if mode in ("db", "dbsm", "dbke"):
            self._assert_compatible(other)
            if mode == "dbke":
                freq_grid = self._frequency_value_to_hz(self.frequencies).reshape(1, 1, -1, 1)
                diff_db = self.linear_to_dbke(self.rcs_power, freq_grid) - self.linear_to_dbke(other.rcs_power, freq_grid)
                diff_linear = np.asarray(self.dbke_to_linear(diff_db, freq_grid), dtype=np.float32)
            else:
                diff_db = self.rcs_to_dbsm(self.rcs_power) - self.rcs_to_dbsm(other.rcs_power)
                diff_linear = np.asarray(10.0 ** (diff_db / 10.0), dtype=np.float32)
            return self._new_grid(
                np.array(self.azimuths, copy=True),
                np.array(self.elevations, copy=True),
                np.array(self.frequencies, copy=True),
                np.array(self.polarizations, copy=True),
                rcs_power=diff_linear,
                rcs_phase=np.full(diff_linear.shape, np.nan, dtype=np.float32),
                rcs_domain="power_phase",
            )
        raise ValueError("mode must be 'coherent', 'incoherent', 'db', or 'dbke'")

    def statistics_dataset(
        self,
        statistic="mean",
        axes=("azimuth", "elevation", "frequency"),
        *,
        domain="magnitude",
        percentile=50.0,
        broadcast_reduced=False,
    ):
        """Compute a statistic over selected axes and return a dataset."""
        axis_map = {"azimuth": 0, "elevation": 1, "frequency": 2, "polarization": 3}
        axis_alias = {
            "azimuths": "azimuth",
            "elevations": "elevation",
            "frequencies": "frequency",
            "polarizations": "polarization",
            "az": "azimuth",
            "el": "elevation",
            "freq": "frequency",
            "pol": "polarization",
        }

        axes_list = self._as_list(axes)
        if axes_list is None:
            raise ValueError("axes must include at least one axis")
        reduce_axes = []
        for axis_name in axes_list:
            key = str(axis_name).strip().lower()
            key = axis_alias.get(key, key)
            if key not in axis_map:
                raise ValueError(f"unknown axis: {axis_name}")
            idx = axis_map[key]
            if idx not in reduce_axes:
                reduce_axes.append(idx)
        if not reduce_axes:
            raise ValueError("axes must include at least one axis")
        reduce_axes = tuple(sorted(reduce_axes))

        if domain == "complex":
            values = self.rcs
        elif domain == "magnitude":
            values = self.rcs_power
        elif domain in ("db", "dbsm"):
            values = self.linear_to_dbsm(self.rcs_power)
        elif domain == "dbke":
            freq_grid = self._frequency_value_to_hz(self.frequencies).reshape(1, 1, -1, 1)
            values = self.linear_to_dbke(self.rcs_power, freq_grid)
        else:
            raise ValueError("domain must be 'complex', 'magnitude', 'dbsm', or 'dbke'")

        stat_key = str(statistic).strip().lower()
        if stat_key.startswith("p") and stat_key[1:].replace(".", "", 1).isdigit():
            percentile = float(stat_key[1:])
            stat_key = "percentile"

        if domain == "complex" and stat_key == "percentile":
            raise ValueError("percentile on complex values is not supported; use magnitude, dbsm, or dbke domain")

        if stat_key == "mean":
            reduced = np.nanmean(values, axis=reduce_axes, keepdims=True)
        elif stat_key == "median":
            reduced = np.nanmedian(values, axis=reduce_axes, keepdims=True)
        elif stat_key == "min":
            reduced = np.nanmin(values, axis=reduce_axes, keepdims=True)
        elif stat_key == "max":
            reduced = np.nanmax(values, axis=reduce_axes, keepdims=True)
        elif stat_key == "std":
            reduced = np.nanstd(values, axis=reduce_axes, keepdims=True)
        elif stat_key == "percentile":
            reduced = np.nanpercentile(values, float(percentile), axis=reduce_axes, keepdims=True)
        else:
            raise ValueError(
                "statistic must be mean, median, min, max, std, percentile, or pXX (for percentile XX)"
            )

        axis_values = [
            np.array(self.azimuths, copy=True),
            np.array(self.elevations, copy=True),
            np.array(self.frequencies, copy=True),
            np.array(self.polarizations, copy=True),
        ]
        if broadcast_reduced:
            # Repeat the reduced result across each reduced axis so the output
            # keeps original axis lengths for downstream plotting.
            reduced = np.broadcast_to(reduced, values.shape).copy()
        else:
            for axis_idx in reduce_axes:
                original = axis_values[axis_idx]
                if axis_idx == 3:
                    axis_values[axis_idx] = np.asarray(["ALL"])
                else:
                    numeric = np.asarray(original, dtype=float)
                    rep = float(np.nanmean(numeric)) if numeric.size else 0.0
                    axis_values[axis_idx] = np.asarray([rep], dtype=float)

        if domain == "complex":
            return self._new_grid(
                axis_values[0],
                axis_values[1],
                axis_values[2],
                axis_values[3],
                reduced,
                rcs_domain="power_phase",
            )
        if domain == "magnitude":
            return self._new_grid(
                axis_values[0],
                axis_values[1],
                axis_values[2],
                axis_values[3],
                rcs_power=np.asarray(reduced, dtype=np.float32),
                rcs_phase=np.full(reduced.shape, np.nan, dtype=np.float32),
                rcs_domain="power_phase",
            )
        # db domain: compute in a log domain, then store as linear so future conversion reproduces the reduced values.
        if domain == "dbke":
            freq_grid = self._frequency_value_to_hz(axis_values[2]).reshape(1, 1, -1, 1)
            reduced_linear = np.asarray(self.dbke_to_linear(np.asarray(reduced, dtype=float), freq_grid), dtype=np.float32)
        else:
            reduced_linear = np.asarray(10.0 ** (np.asarray(reduced, dtype=float) / 10.0), dtype=np.float32)
        return self._new_grid(
            axis_values[0],
            axis_values[1],
            axis_values[2],
            axis_values[3],
            rcs_power=reduced_linear,
            rcs_phase=np.full(reduced_linear.shape, np.nan, dtype=np.float32),
            rcs_domain="power_phase",
        )

    def _index_for_value(self, axis, value, tol=0.0):
        """Find the first index of a value on an axis.

        Args:
            axis: 1D array to search.
            value: Value to find.
            tol: Absolute tolerance for numeric matching.

        Returns:
            Integer index of the first match.

        Raises:
            ValueError: if no match is found.
        """
        axis_arr = np.asarray(axis)
        if tol > 0.0:
            matches = np.where(np.isclose(axis_arr, value, atol=tol, rtol=0.0))[0]
        else:
            matches = np.where(axis_arr == value)[0]
        if matches.size == 0:
            raise ValueError(f"value {value} not found on axis")
        return int(matches[0])

    def get_by_value(self, azimuth, elevation, frequency, polarization, tol=0.0):
        """Fetch a single sample by axis values.

        Use when you have physical axis values rather than indices.

        Args:
            azimuth: Azimuth value.
            elevation: Elevation value.
            frequency: Frequency value.
            polarization: Polarization label.
            tol: Absolute tolerance for numeric matching.

        Returns:
            Complex RCS sample.
        """
        az_idx = self._index_for_value(self.azimuths, azimuth, tol=tol)
        el_idx = self._index_for_value(self.elevations, elevation, tol=tol)
        f_idx = self._index_for_value(self.frequencies, frequency, tol=tol)
        p_idx = self._index_for_value(self.polarizations, polarization, tol=tol)
        return self.rcs[az_idx, el_idx, f_idx, p_idx]

    def rcs_to_dbsm(self, rcs_value, eps=1e-12):
        """Convert linear RCS to dBsm.

        Args:
            rcs_value: Complex or real RCS value(s).
            eps: Floor to avoid log(0).

        Returns:
            dBsm value(s) as float or ndarray.
        """
        linear = self.rcs_to_linear(rcs_value)
        return self.linear_to_dbsm(linear, eps=eps)

    def rcs_to_dbke(self, rcs_value, frequency_value, eps=1e-12):
        """Convert linear 2D scattering width to absolute dBke."""
        linear = self.rcs_to_linear(rcs_value)
        return self.linear_to_dbke(linear, frequency_value, eps=eps)

    def rcs_to_display_db(self, rcs_value, frequency_value=None, eps=1e-12):
        """Convert to the dataset's preferred log-power display unit."""
        linear = self.rcs_to_linear(rcs_value)
        return self.linear_to_default_db(linear, frequency_value=frequency_value, eps=eps)

    def get_dbsm(self, az_idx, el_idx, f_idx, p_idx, eps=1e-12):
        """Fetch a sample by indices and return dBsm."""
        return self.linear_to_dbsm(self.rcs_power[az_idx, el_idx, f_idx, p_idx], eps=eps)

    def get_dbke(self, az_idx, el_idx, f_idx, p_idx, eps=1e-12):
        """Fetch a sample by indices and return dBke."""
        freq_value = self.frequencies[f_idx]
        return self.linear_to_dbke(self.rcs_power[az_idx, el_idx, f_idx, p_idx], freq_value, eps=eps)

    def get_dbsm_by_value(self, azimuth, elevation, frequency, polarization, tol=0.0, eps=1e-12):
        """Fetch a sample by axis values and return dBsm."""
        az_idx = self._index_for_value(self.azimuths, azimuth, tol=tol)
        el_idx = self._index_for_value(self.elevations, elevation, tol=tol)
        f_idx = self._index_for_value(self.frequencies, frequency, tol=tol)
        p_idx = self._index_for_value(self.polarizations, polarization, tol=tol)
        return self.linear_to_dbsm(self.rcs_power[az_idx, el_idx, f_idx, p_idx], eps=eps)

    def get_dbke_by_value(self, azimuth, elevation, frequency, polarization, tol=0.0, eps=1e-12):
        """Fetch a sample by axis values and return dBke."""
        az_idx = self._index_for_value(self.azimuths, azimuth, tol=tol)
        el_idx = self._index_for_value(self.elevations, elevation, tol=tol)
        f_idx = self._index_for_value(self.frequencies, frequency, tol=tol)
        p_idx = self._index_for_value(self.polarizations, polarization, tol=tol)
        return self.linear_to_dbke(self.rcs_power[az_idx, el_idx, f_idx, p_idx], self.frequencies[f_idx], eps=eps)

    def save(self, path):
        """Save the grid to a .grim (npz) file.

        Args:
            path: Output path, with or without .grim.

        Returns:
            The actual path written (always ends with .grim).
        """
        if not path.endswith(".grim"):
            path = f"{path}.grim"
        with open(path, "wb") as f:
            units_payload = json.dumps(self.units) if self.units else ""
            np.savez(
                f,
                azimuths=self.azimuths,
                elevations=self.elevations,
                frequencies=self.frequencies,
                polarizations=self.polarizations,
                rcs_power=self.rcs_power.astype(np.float32),
                rcs_phase=self.rcs_phase.astype(np.float32),
                rcs_domain="power_phase",
                power_domain=self.power_domain,
                source_path=self.source_path if self.source_path is not None else "",
                history=self.history if self.history is not None else "",
                units=units_payload,
            )
        return path

    @classmethod
    def load(cls, path, mmap_mode: str | None = None):
        """Load a grid from a .grim (npz) file.

        Args:
            path: Input path, with or without .grim.
            mmap_mode: Optional numpy mmap mode (e.g., "r") for lazy loading.

        Returns:
            RcsGrid instance loaded from disk.
        """
        if not path.endswith(".grim"):
            path = f"{path}.grim"
        with open(path, "rb") as f:
            data = np.load(f, mmap_mode=mmap_mode, allow_pickle=False)

            units = {}
            if "units" in data:
                raw_units = data["units"]
                if isinstance(raw_units, np.ndarray):
                    raw_units = raw_units.item()
                if isinstance(raw_units, bytes):
                    raw_units = raw_units.decode("utf-8")
                if isinstance(raw_units, str) and raw_units:
                    try:
                        units = json.loads(raw_units)
                    except json.JSONDecodeError:
                        units = {}
                elif isinstance(raw_units, dict):
                    units = raw_units

            source_path_raw = data["source_path"].item() if "source_path" in data else None
            source_path = source_path_raw if source_path_raw else None
            history_raw = data["history"].item() if "history" in data else None
            history = history_raw if history_raw else None
            required = ("azimuths", "elevations", "frequencies", "polarizations", "rcs_power", "rcs_phase")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(
                    f"{path} is not a supported .grim file (missing keys: {', '.join(missing)})"
                )

            return cls(
                data["azimuths"],
                data["elevations"],
                data["frequencies"],
                data["polarizations"],
                rcs_power=data["rcs_power"],
                rcs_phase=data["rcs_phase"],
                rcs_domain="power_phase",
                source_path=source_path,
                history=history,
                units=units,
            )

    @classmethod
    def load_out(cls, path):
        """Load whitespace-delimited `.out` data into an RcsGrid.

        Expected columns per non-comment line:
            frequency_ghz  azimuth_deg  rcs_dbke  phase_deg

        Parsing rules:
            - Lines starting with `#` (or text after `#`) are ignored.
            - Values are whitespace-delimited.
            - Polarization is inferred from filename (`HH` or `VV`);
              if not present, polarization is `NA`.
            - The third column is interpreted as absolute dBke and converted to
              linear 2D scattering width using sigma_2d = (lambda / 2pi) * 10^(dBke/10).

        Output mapping:
            - azimuth axis   <- angle column
            - elevation axis <- single value [0.0]
            - frequency axis <- frequency_ghz column
            - polarization   <- inferred filename polarization
            - stored power   <- linear 2D scattering width (matches .grim storage)
        """

        file_name = os.path.basename(str(path))
        stem_upper = os.path.splitext(file_name)[0].upper()
        pol_match = re.search(r"(?<![A-Z0-9])(HH|VV)(?![A-Z0-9])", stem_upper)
        if pol_match is not None:
            pol_label = pol_match.group(1)
        else:
            idx_hh = stem_upper.find("HH")
            idx_vv = stem_upper.find("VV")
            if idx_hh < 0 and idx_vv < 0:
                pol_label = "NA"
            elif idx_hh >= 0 and (idx_vv < 0 or idx_hh <= idx_vv):
                pol_label = "HH"
            else:
                pol_label = "VV"

        records: list[tuple[float, float, float, float]] = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(
                        f"line {line_no}: expected 4 columns "
                        "(frequency_ghz azimuth_deg rcs_dbke phase_deg)"
                    )
                try:
                    freq_ghz = float(parts[0])
                    azimuth_deg = float(parts[1])
                    rcs_dbke = float(parts[2])
                    phase_deg = float(parts[3])
                except ValueError as exc:
                    raise ValueError(f"line {line_no}: invalid numeric value ({exc})") from exc

                if not (np.isfinite(freq_ghz) and np.isfinite(azimuth_deg)):
                    continue
                records.append((freq_ghz, azimuth_deg, rcs_dbke, phase_deg))

        if not records:
            raise ValueError("OUT contains no data rows")

        frequencies = np.asarray(sorted({r[0] for r in records}), dtype=float)
        azimuths = np.asarray(sorted({r[1] for r in records}), dtype=float)
        elevations = np.asarray([0.0], dtype=float)
        polarizations = np.asarray([pol_label], dtype=object)

        f_idx = {float(v): i for i, v in enumerate(frequencies.tolist())}
        az_idx = {float(v): i for i, v in enumerate(azimuths.tolist())}

        shape = (len(azimuths), 1, len(frequencies), 1)
        power = np.full(shape, np.nan, dtype=np.float32)
        phase = np.full(shape, np.nan, dtype=np.float32)

        for freq_ghz, azimuth_deg, rcs_dbke, phase_deg in records:
            ai = az_idx[float(azimuth_deg)]
            fi = f_idx[float(freq_ghz)]
            if np.isfinite(rcs_dbke):
                lambda_m = C0 / (float(freq_ghz) * 1.0e9) if float(freq_ghz) > 0.0 else float("nan")
                sigma_2d = (lambda_m / (2.0 * np.pi)) * (10.0 ** (rcs_dbke / 10.0)) if np.isfinite(lambda_m) else float("nan")
                power[ai, 0, fi, 0] = np.float32(sigma_2d)
            else:
                power[ai, 0, fi, 0] = np.nan
            if np.isfinite(phase_deg):
                phase[ai, 0, fi, 0] = np.float32(np.deg2rad(phase_deg))
            else:
                phase[ai, 0, fi, 0] = np.nan

        if not np.isfinite(power).any():
            raise ValueError("OUT parsed, but no finite RCS magnitude values were found")

        return cls(
            azimuths,
            elevations,
            frequencies,
            polarizations,
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain="power_phase",
            source_path=path,
            history=f"Loaded OUT (dBke -> linear sigma_2d): {path}",
            units={"azimuth": "deg", "elevation": "deg", "frequency": "GHz", "rcs_log_unit": "dBke"},
        )

    @classmethod
    def load_theta_phi_csv(cls, path):
        """Load a theta/phi scattering CSV into an RcsGrid.

        Expected layout:
            - Two header rows total (or any leading metadata rows), with one row
              containing column names like:
              frequency(hz), theta(deg), phi(deg),
              rcs theta-theta(dbsm), rcs phi-theta(dbsm),
              rcs theta-phi(dbsm), rcs phi-phi,
              phase theta-theta(...), phase phi-theta(...),
              phase theta-phi(...), phase phi-phi(...)

        Conventions applied:
            - phi(deg)   -> azimuth axis
            - theta(deg) -> elevation axis
            - theta -> V, phi -> H
              rcs theta-theta -> VV
              rcs phi-theta   -> HV
              rcs theta-phi   -> VH
              rcs phi-phi     -> HH
            - RCS columns are interpreted as dBsm and converted to linear power.
            - Phase columns are interpreted as degrees and converted to radians.
        """

        def _norm(text: str) -> str:
            s = str(text).strip().lower()
            for ch in (" ", "_", "\t"):
                s = s.replace(ch, "")
            return s

        def _infer_freq_scale_to_ghz(freq_header_token: str, freq_values: np.ndarray) -> tuple[float, str]:
            token = str(freq_header_token or "").lower()
            if "ghz" in token:
                return 1.0, "GHz"
            if "mhz" in token:
                return 1.0e-3, "MHz"
            if "khz" in token:
                return 1.0e-6, "kHz"
            if "hz" in token:
                return 1.0e-9, "Hz"

            finite = np.asarray(freq_values, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size == 0:
                return 1.0, "GHz"

            typical = float(np.nanmedian(np.abs(finite)))
            if typical >= 1.0e6:
                return 1.0e-9, "Hz"
            if typical >= 1.0e3:
                return 1.0e-3, "MHz"
            return 1.0, "GHz"

        alias_to_key = {
            "frequency(hz)": "frequency",
            "frequencyhz": "frequency",
            "frequency(ghz)": "frequency",
            "frequencyghz": "frequency",
            "frequency(mhz)": "frequency",
            "frequencymhz": "frequency",
            "frequency(khz)": "frequency",
            "frequencykhz": "frequency",
            "frequency": "frequency",
            "theta(deg)": "theta_deg",
            "theta": "theta_deg",
            "phi(deg)": "phi_deg",
            "phi": "phi_deg",
            "rcstheta-theta(dbsm)": "rcs_vv_dbsm",
            "rcstheta-thetadbsm": "rcs_vv_dbsm",
            "rcstheta-theta(dbm^2)": "rcs_vv_dbsm",
            "rcstheta-thetadbm2": "rcs_vv_dbsm",
            "rcstheta-theta": "rcs_vv_dbsm",
            "rcsphi-theta(dbsm)": "rcs_hv_dbsm",
            "rcsphi-thetadbsm": "rcs_hv_dbsm",
            "rcsphi-theta(dbm^2)": "rcs_hv_dbsm",
            "rcsphi-thetadbm2": "rcs_hv_dbsm",
            "rcsphi-theta": "rcs_hv_dbsm",
            "rcstheta-phi(dbsm)": "rcs_vh_dbsm",
            "rcstheta-phidbsm": "rcs_vh_dbsm",
            "rcstheta-phi(dbm^2)": "rcs_vh_dbsm",
            "rcstheta-phidbm2": "rcs_vh_dbsm",
            "rcstheta-phi": "rcs_vh_dbsm",
            "rcsphi-phi(dbsm)": "rcs_hh_dbsm",
            "rcsphi-phidbsm": "rcs_hh_dbsm",
            "rcsphi-phi(dbm^2)": "rcs_hh_dbsm",
            "rcsphi-phidbm2": "rcs_hh_dbsm",
            "rcsphi-phi": "rcs_hh_dbsm",
            "phasetheta-theta(deg)": "phase_vv_deg",
            "phasetheta-theta(dbsm)": "phase_vv_deg",
            "phasetheta-theta": "phase_vv_deg",
            "phasephi-theta(deg)": "phase_hv_deg",
            "phasephi-theta(dbsm)": "phase_hv_deg",
            "phasephi-theta": "phase_hv_deg",
            "phasetheta-phi(deg)": "phase_vh_deg",
            "phasetheta-phi(dbsm)": "phase_vh_deg",
            "phasetheta-phi": "phase_vh_deg",
            "phasephi-phi(deg)": "phase_hh_deg",
            "phasephi-phi(dbsm)": "phase_hh_deg",
            "phasephi-phi": "phase_hh_deg",
        }

        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        if not rows:
            raise ValueError("CSV is empty")

        def _classify_fuzzy_header(cell_value: str) -> str | None:
            raw = str(cell_value or "").strip().lower()
            if raw == "":
                return None

            key = alias_to_key.get(_norm(raw))
            if key is not None:
                return key

            compact = re.sub(r"[^a-z0-9]+", "", raw)
            if compact in {"f", "freq"} or "frequency" in compact:
                return "frequency"
            if (
                "theta" in compact
                and "phase" not in compact
                and "rcs" not in compact
                and "abs" not in compact
            ):
                return "theta_deg"
            if (
                "phi" in compact
                and "phase" not in compact
                and "rcs" not in compact
                and "abs" not in compact
            ):
                return "phi_deg"

            has_phase = "phase" in compact
            has_mag = (("rcs" in compact) or ("abs" in compact) or ("sigma" in compact)) and not has_phase
            if not has_phase and not has_mag:
                return None

            pair_key: str | None = None
            theta_count = len(re.findall("theta", raw))
            phi_count = len(re.findall("phi", raw))
            if "phi-theta" in raw or re.search(r"phi[^a-z0-9]+theta", raw):
                pair_key = "hv"
            elif "theta-phi" in raw or re.search(r"theta[^a-z0-9]+phi", raw):
                pair_key = "vh"
            elif theta_count >= 2:
                pair_key = "vv"
            elif phi_count >= 2:
                pair_key = "hh"
            elif theta_count == 1 and phi_count == 0:
                pair_key = "vv"
            elif phi_count == 1 and theta_count == 0:
                pair_key = "hh"
            elif theta_count == 1 and phi_count == 1:
                pair_key = "hv" if raw.find("phi") < raw.find("theta") else "vh"

            if pair_key is None:
                return None
            if has_phase:
                return f"phase_{pair_key}_deg"
            return f"rcs_{pair_key}_dbsm"

        def _is_numeric_axes_row(row_values: list[str]) -> bool:
            if len(row_values) < 3:
                return False
            try:
                f_val = float(str(row_values[0]).strip())
                t_val = float(str(row_values[1]).strip())
                p_val = float(str(row_values[2]).strip())
            except ValueError:
                return False
            return bool(np.isfinite(f_val) and np.isfinite(t_val) and np.isfinite(p_val))

        header_idx = None
        data_start_idx = 0
        col_idx: dict[str, int] = {}
        header_tokens: dict[str, str] = {}
        required_axes = {"frequency", "theta_deg", "phi_deg"}
        for i, row in enumerate(rows):
            mapped: dict[str, int] = {}
            mapped_tokens: dict[str, str] = {}
            for j, cell in enumerate(row):
                key = _classify_fuzzy_header(cell)
                if key is not None and key not in mapped:
                    mapped[key] = j
                    mapped_tokens[key] = str(cell)
            has_any_rcs = any(k.startswith("rcs_") for k in mapped.keys())
            if required_axes.issubset(mapped.keys()) and has_any_rcs:
                header_idx = i
                data_start_idx = i + 1
                col_idx = mapped
                header_tokens = mapped_tokens
                break

        if header_idx is None:
            for i, row in enumerate(rows):
                if _is_numeric_axes_row(row):
                    header_idx = i - 1
                    data_start_idx = i
                    col_idx = {"frequency": 0, "theta_deg": 1, "phi_deg": 2}
                    if len(row) > 3:
                        col_idx["rcs_vv_dbsm"] = 3
                    if len(row) > 4:
                        col_idx["rcs_hv_dbsm"] = 4
                    if len(row) > 5:
                        col_idx["rcs_vh_dbsm"] = 5
                    if len(row) > 6:
                        col_idx["rcs_hh_dbsm"] = 6
                    if len(row) > 7:
                        col_idx["phase_vv_deg"] = 7
                    if len(row) > 8:
                        col_idx["phase_hv_deg"] = 8
                    if len(row) > 9:
                        col_idx["phase_vh_deg"] = 9
                    if len(row) > 10:
                        col_idx["phase_hh_deg"] = 10
                    break

        if "frequency" not in col_idx or "theta_deg" not in col_idx or "phi_deg" not in col_idx:
            raise ValueError(
                "Could not find CSV axes. Need frequency/theta/phi columns (header-based or first 3 columns)."
            )

        def _parse_float(raw: str) -> float:
            text = str(raw).strip()
            if text == "":
                return float("nan")
            return float(text)

        records: list[tuple[float, float, float, float, float, float, float, float, float, float, float]] = []
        for row in rows[data_start_idx:]:
            if not row or all(str(cell).strip() == "" for cell in row):
                continue
            try:
                f_hz = _parse_float(row[col_idx["frequency"]]) if col_idx["frequency"] < len(row) else float("nan")
                theta_deg = _parse_float(row[col_idx["theta_deg"]]) if col_idx["theta_deg"] < len(row) else float("nan")
                phi_deg = _parse_float(row[col_idx["phi_deg"]]) if col_idx["phi_deg"] < len(row) else float("nan")
            except ValueError:
                # Skip non-numeric rows after the header.
                continue

            if not (np.isfinite(f_hz) and np.isfinite(theta_deg) and np.isfinite(phi_deg)):
                continue

            def _cell(key: str) -> float:
                idx = col_idx.get(key, -1)
                if idx < 0 or idx >= len(row):
                    return float("nan")
                try:
                    return _parse_float(row[idx])
                except ValueError:
                    return float("nan")

            records.append(
                (
                    float(f_hz),
                    float(theta_deg),
                    float(phi_deg),
                    _cell("rcs_vv_dbsm"),
                    _cell("rcs_hv_dbsm"),
                    _cell("rcs_vh_dbsm"),
                    _cell("rcs_hh_dbsm"),
                    _cell("phase_vv_deg"),
                    _cell("phase_hv_deg"),
                    _cell("phase_vh_deg"),
                    _cell("phase_hh_deg"),
                )
            )

        if not records:
            raise ValueError("CSV contains no data rows after the header")

        raw_freqs = np.asarray([r[0] for r in records], dtype=float)
        freq_scale_to_ghz, _ = _infer_freq_scale_to_ghz(header_tokens.get("frequency", ""), raw_freqs)
        records_ghz = [
            (
                float(f_raw * freq_scale_to_ghz),
                theta_deg,
                phi_deg,
                vv_db,
                hv_db,
                vh_db,
                hh_db,
                vv_ph,
                hv_ph,
                vh_ph,
                hh_ph,
            )
            for (
                f_raw,
                theta_deg,
                phi_deg,
                vv_db,
                hv_db,
                vh_db,
                hh_db,
                vv_ph,
                hv_ph,
                vh_ph,
                hh_ph,
            ) in records
        ]

        freqs = np.asarray(sorted({r[0] for r in records_ghz}), dtype=float)
        elevs = np.asarray(sorted({r[1] for r in records}), dtype=float)   # theta -> elevation
        azims = np.asarray(sorted({r[2] for r in records}), dtype=float)   # phi -> azimuth
        pols = np.asarray(["VV", "HV", "VH", "HH"], dtype=object)

        f_idx = {float(v): i for i, v in enumerate(freqs.tolist())}
        el_idx = {float(v): i for i, v in enumerate(elevs.tolist())}
        az_idx = {float(v): i for i, v in enumerate(azims.tolist())}

        shape = (len(azims), len(elevs), len(freqs), len(pols))
        power = np.full(shape, np.nan, dtype=np.float32)
        phase = np.full(shape, np.nan, dtype=np.float32)

        def _dbsm_to_linear(value: float) -> float:
            if not np.isfinite(value):
                return float("nan")
            return float(10.0 ** (value / 10.0))

        def _deg_to_rad(value: float) -> float:
            if not np.isfinite(value):
                return float("nan")
            return float(np.deg2rad(value))

        for (
            f_ghz,
            theta_deg,
            phi_deg,
            vv_db,
            hv_db,
            vh_db,
            hh_db,
            vv_ph,
            hv_ph,
            vh_ph,
            hh_ph,
        ) in records_ghz:
            ai = az_idx[phi_deg]
            ei = el_idx[theta_deg]
            fi = f_idx[f_ghz]

            power[ai, ei, fi, 0] = _dbsm_to_linear(vv_db)  # VV (theta-theta)
            power[ai, ei, fi, 1] = _dbsm_to_linear(hv_db)  # HV (phi-theta)
            power[ai, ei, fi, 2] = _dbsm_to_linear(vh_db)  # VH (theta-phi)
            power[ai, ei, fi, 3] = _dbsm_to_linear(hh_db)  # HH (phi-phi)

            phase[ai, ei, fi, 0] = _deg_to_rad(vv_ph)
            phase[ai, ei, fi, 1] = _deg_to_rad(hv_ph)
            phase[ai, ei, fi, 2] = _deg_to_rad(vh_ph)
            phase[ai, ei, fi, 3] = _deg_to_rad(hh_ph)

        if not np.isfinite(power).any():
            raise ValueError("CSV parsed, but no finite RCS magnitude values were found")

        return cls(
            azims,
            elevs,
            freqs,
            pols,
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain="power_phase",
            source_path=path,
            history=f"Loaded theta/phi CSV: {path}",
            units={"azimuth": "deg", "elevation": "deg", "frequency": "GHz"},
        )

    @classmethod
    def load_theta_phi_txt(cls, path):
        """Load whitespace-delimited theta/phi TXT format into an RcsGrid.

        Expected columns after two header rows:
            theta(deg), phi(deg), abs(rcs)(dbm^2), abs(theta)(dbm^2),
            phase(theta)(deg), abs(phi)(dbm^2), phase(phi)(deg), ax.ratio(db)

        Axis/polarization mapping:
            - theta(deg) -> azimuth
            - phi(deg)   -> elevation
            - theta -> V, phi -> H
              abs(theta), phase(theta) -> VV
              abs(phi),   phase(phi)   -> HH
            - abs(rcs) is loaded as a third polarization channel: TOTAL
        """

        def _norm_token(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())

        def _frequency_from_filename_ghz(file_path: str) -> float | None:
            name = os.path.basename(str(file_path))
            match = re.search(
                r"(?:^|[^a-z0-9])f\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-z]+)?",
                name,
                flags=re.IGNORECASE,
            )
            if match is None:
                return None
            try:
                raw_value = float(match.group(1))
            except (TypeError, ValueError):
                return None
            if not np.isfinite(raw_value):
                return None

            raw_unit = (match.group(2) or "").strip().lower()
            unit = raw_unit
            if unit.startswith("ghz"):
                scale = 1.0
            elif unit.startswith("mhz"):
                scale = 1.0e-3
            elif unit.startswith("khz"):
                scale = 1.0e-6
            elif unit.startswith("hz"):
                scale = 1.0e-9
            else:
                magnitude = abs(raw_value)
                if magnitude >= 1.0e6:
                    scale = 1.0e-9
                elif magnitude >= 1.0e3:
                    scale = 1.0e-3
                else:
                    scale = 1.0
            return float(raw_value * scale)

        alias_to_key = {
            "thetadeg": "theta_deg",
            "phideg": "phi_deg",
            "absrcsdbm2": "abs_rcs_dbm2",
            "absthetadbm2": "abs_theta_dbm2",
            "phasethetadeg": "phase_theta_deg",
            "absphidbm2": "abs_phi_dbm2",
            "phasephideg": "phase_phi_deg",
            "axratiodb": "ax_ratio_db",
        }

        with open(path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()
        if not lines:
            raise ValueError("TXT is empty")

        header_idx = None
        data_start_idx = 0
        col_idx: dict[str, int] = {}
        required = {
            "theta_deg",
            "phi_deg",
            "abs_theta_dbm2",
            "phase_theta_deg",
            "abs_phi_dbm2",
            "phase_phi_deg",
        }

        fallback_col_idx = {
            "theta_deg": 0,
            "phi_deg": 1,
            "abs_rcs_dbm2": 2,
            "abs_theta_dbm2": 3,
            "phase_theta_deg": 4,
            "abs_phi_dbm2": 5,
            "phase_phi_deg": 6,
            "ax_ratio_db": 7,
        }

        def _tokenize(text: str) -> list[str]:
            return [tok for tok in re.split(r"[,\s]+", text.strip()) if tok]

        def _is_numeric_data_line(tokens: list[str]) -> bool:
            if len(tokens) < 7:
                return False
            numeric_needed = (0, 1, 3, 4, 5, 6)
            for idx in numeric_needed:
                if idx >= len(tokens):
                    return False
                try:
                    float(tokens[idx])
                except ValueError:
                    return False
            return True

        for i, line in enumerate(lines):
            tokens = _tokenize(line)
            mapped: dict[str, int] = {}
            for j, token in enumerate(tokens):
                key = alias_to_key.get(_norm_token(token))
                if key is not None and key not in mapped:
                    mapped[key] = j
            if required.issubset(mapped.keys()):
                header_idx = i
                col_idx = mapped
                data_start_idx = i + 1
                break

        if header_idx is None:
            for i, line in enumerate(lines):
                tokens = _tokenize(line)
                if _is_numeric_data_line(tokens):
                    col_idx = dict(fallback_col_idx)
                    data_start_idx = i
                    break
            else:
                raise ValueError(
                    "Could not parse TXT: expected header columns or numeric rows with at least 7 columns."
                )

        def _parse_float(raw: str) -> float:
            text = str(raw).strip()
            if text == "":
                return float("nan")
            return float(text)

        records: list[tuple[float, float, float, float, float, float, float, float]] = []
        for line in lines[data_start_idx:]:
            tokens = _tokenize(line)
            if not tokens:
                continue

            def _cell(key: str) -> float:
                idx = col_idx.get(key, -1)
                if idx < 0 or idx >= len(tokens):
                    return float("nan")
                try:
                    return _parse_float(tokens[idx])
                except ValueError:
                    return float("nan")

            theta_deg = _cell("theta_deg")
            phi_deg = _cell("phi_deg")
            if not (np.isfinite(theta_deg) and np.isfinite(phi_deg)):
                continue

            records.append(
                (
                    float(theta_deg),
                    float(phi_deg),
                    _cell("abs_theta_dbm2"),
                    _cell("phase_theta_deg"),
                    _cell("abs_phi_dbm2"),
                    _cell("phase_phi_deg"),
                    _cell("abs_rcs_dbm2"),
                    _cell("ax_ratio_db"),
                )
            )

        if not records:
            raise ValueError("TXT contains no data rows after header")

        azims = np.asarray(sorted({r[0] for r in records}), dtype=float)   # theta -> azimuth
        elevs = np.asarray(sorted({r[1] for r in records}), dtype=float)   # phi -> elevation
        freq_ghz = _frequency_from_filename_ghz(path)
        if freq_ghz is None:
            freqs = np.asarray([0.0], dtype=float)
            freq_unit = "arb"
        else:
            freqs = np.asarray([float(freq_ghz)], dtype=float)
            freq_unit = "GHz"
        pols = np.asarray(["VV", "HH", "TOTAL"], dtype=object)

        el_idx = {float(v): i for i, v in enumerate(elevs.tolist())}
        az_idx = {float(v): i for i, v in enumerate(azims.tolist())}

        shape = (len(azims), len(elevs), 1, len(pols))
        power = np.full(shape, np.nan, dtype=np.float32)
        phase = np.full(shape, np.nan, dtype=np.float32)

        def _db_to_linear(value: float) -> float:
            if not np.isfinite(value):
                return float("nan")
            return float(10.0 ** (value / 10.0))

        def _deg_to_rad(value: float) -> float:
            if not np.isfinite(value):
                return float("nan")
            return float(np.deg2rad(value))

        for theta_deg, phi_deg, abs_theta_db, ph_theta_deg, abs_phi_db, ph_phi_deg, abs_rcs_db, _ in records:
            ai = az_idx[theta_deg]
            ei = el_idx[phi_deg]
            power[ai, ei, 0, 0] = _db_to_linear(abs_theta_db)   # VV
            phase[ai, ei, 0, 0] = _deg_to_rad(ph_theta_deg)
            power[ai, ei, 0, 1] = _db_to_linear(abs_phi_db)     # HH
            phase[ai, ei, 0, 1] = _deg_to_rad(ph_phi_deg)
            power[ai, ei, 0, 2] = _db_to_linear(abs_rcs_db)     # TOTAL

        if not np.isfinite(power).any():
            raise ValueError("TXT parsed, but no finite magnitude values were found")

        return cls(
            azims,
            elevs,
            freqs,
            pols,
            rcs_power=power,
            rcs_phase=phase,
            rcs_domain="power_phase",
            source_path=path,
            history=f"Loaded theta/phi TXT: {path}",
            units={"azimuth": "deg", "elevation": "deg", "frequency": freq_unit},
        )

    @classmethod
    def load_pio(cls, path):
        """Load a Pioneer (.pio / .cmplx_di) file into an RcsGrid.

        File layout:
            - ASCII header of `key=value` lines, terminated by a line whose
              key is `Offset` (giving the byte offset of the binary block).
            - Binary block of interleaved real/imag floats (single or double
              precision per the `precision` header field) of length
              xsize*ysize*2.
            - Optional ASCII footer of `key=value` lines (e.g. polarity, log).

        Axis convention (this loader):
            - X axis (xname=azimuth/position) -> azimuth
            - Y axis (yname=frequency)        -> frequency, scaled to GHz from
              yunits in {Hz, kHz, MHz, GHz}
            - elevation axis is a single 0.0
            - polarization is taken from the `polarity` header/footer field, or
              inferred from HH/VV/VH/HV in the filename.
        """
        header: dict[str, str] = {}
        footer: dict[str, str] = {}
        first_line: str = ""

        with open(path, "rb") as f:
            raw_first = f.readline()
            first_line = raw_first.decode("ascii", errors="replace").strip()

            # Read header until a line with key 'offset' (case-insensitive).
            while True:
                raw_line = f.readline()
                if not raw_line:
                    raise ValueError("Unexpected EOF while reading PIO header")
                line = raw_line.decode("ascii", errors="replace").strip()
                if "=" in line:
                    key, _, value = line.partition("=")
                    key_l = key.strip().lower()
                    header[key_l] = value.strip()
                    if key_l == "offset":
                        break

            offset_raw = header.get("offset")
            if offset_raw is None:
                raise ValueError("PIO header missing 'Offset='")
            try:
                offset = int(float(offset_raw))
            except ValueError as exc:
                raise ValueError(f"PIO header has non-numeric Offset: {offset_raw!r}") from exc

            def _int(key: str) -> int | None:
                raw = header.get(key)
                if raw is None:
                    return None
                try:
                    return int(float(raw))
                except ValueError:
                    return None

            xsize = _int("xsize")
            ysize = _int("ysize")
            if xsize is None or ysize is None:
                raise ValueError("PIO header missing xsize/ysize")

            precision = (header.get("precision") or "").strip().lower()
            data_type = (header.get("type") or "complex").strip().lower()

            if precision == "single":
                dtype = np.float32
            elif precision == "double":
                dtype = np.float64
            else:
                raise ValueError(f"Unsupported PIO precision: {precision!r}")

            n_floats = int(xsize) * int(ysize) * (2 if data_type == "complex" else 1)
            itemsize = np.dtype(dtype).itemsize

            f.seek(offset, 0)
            raw_buf = f.read(n_floats * itemsize)
            if len(raw_buf) < n_floats * itemsize:
                raise ValueError(
                    f"PIO data block truncated: expected {n_floats * itemsize} bytes, got {len(raw_buf)}"
                )
            rawdata = np.frombuffer(raw_buf, dtype=dtype, count=n_floats)

            # Anything after the data block is treated as the optional footer.
            footer_blob = f.read()

        for raw_line in footer_blob.splitlines():
            line = raw_line.decode("ascii", errors="replace").strip() if isinstance(raw_line, bytes) else str(raw_line).strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            footer[key.strip().lower()] = value.strip()

        def _parse_axis_values(key: str, expected_size: int) -> np.ndarray | None:
            raw = header.get(key)
            if raw is None:
                return None
            tokens = re.split(r"[:\s,]+", raw.strip())
            values: list[float] = []
            for tok in tokens:
                if not tok:
                    continue
                try:
                    values.append(float(tok))
                except ValueError:
                    return None
            if len(values) == expected_size:
                return np.asarray(values, dtype=float)
            return None

        def _build_axis(prefix: str, size: int) -> np.ndarray:
            vals = _parse_axis_values(f"{prefix}vals", size)
            if vals is not None:
                return vals
            start = header.get(f"{prefix}start")
            stop = header.get(f"{prefix}stop")
            step = header.get(f"{prefix}step")
            try:
                start_f = float(start) if start is not None else None
                stop_f = float(stop) if stop is not None else None
                step_f = float(step) if step is not None else None
            except ValueError:
                start_f = stop_f = step_f = None
            if start_f is not None and step_f is not None:
                return start_f + np.arange(size, dtype=float) * step_f
            if start_f is not None and stop_f is not None and size > 1:
                return np.linspace(start_f, stop_f, size)
            if size == 1 and start_f is not None:
                return np.asarray([start_f], dtype=float)
            raise ValueError(f"Could not reconstruct {prefix} axis from PIO header")

        xvals = _build_axis("x", int(xsize))
        yvals = _build_axis("y", int(ysize))

        xname = (header.get("xname") or "").strip().lower()
        yname = (header.get("yname") or "").strip().lower()
        yunits = (header.get("yunits") or "").strip().lower()

        if data_type == "complex":
            complex_arr = rawdata[0::2].astype(np.float64) + 1j * rawdata[1::2].astype(np.float64)
        else:
            complex_arr = rawdata.astype(np.complex128)

        # MATLAB reshape(data, xsize, ysize) is column-major.
        data_2d = np.asarray(complex_arr, dtype=np.complex64).reshape((int(xsize), int(ysize)), order="F")

        if not (xname in ("azimuth", "position") or yname == "frequency"):
            raise ValueError(
                f"Unsupported PIO axes (xname={xname!r}, yname={yname!r}); "
                "expected azimuth/position vs frequency"
            )

        if yunits == "ghz" or yunits == "":
            freqs_ghz = np.asarray(yvals, dtype=float)
        elif yunits == "mhz":
            freqs_ghz = np.asarray(yvals, dtype=float) * 1.0e-3
        elif yunits == "khz":
            freqs_ghz = np.asarray(yvals, dtype=float) * 1.0e-6
        elif yunits == "hz":
            freqs_ghz = np.asarray(yvals, dtype=float) * 1.0e-9
        else:
            freqs_ghz = np.asarray(yvals, dtype=float)

        pol = (header.get("polarity") or footer.get("polarity") or "").strip().upper()
        if not pol:
            stem = os.path.splitext(os.path.basename(str(path)))[0].upper()
            for tag in ("HH", "VV", "VH", "HV"):
                if tag in stem:
                    pol = tag
                    break
        if not pol:
            pol = "NA"

        azimuths = np.asarray(xvals, dtype=float)
        elevations = np.asarray([0.0], dtype=float)
        polarizations = np.asarray([pol], dtype=object)

        rcs_arr = data_2d[:, np.newaxis, :, np.newaxis].astype(np.complex64)

        prior_log = header.get("log") or footer.get("log") or ""
        history_parts = [f"Loaded Pioneer file: {path}"]
        if prior_log:
            history_parts.append(f"prior log: {prior_log}")
        history = " | ".join(history_parts)

        return cls(
            azimuths,
            elevations,
            freqs_ghz,
            polarizations,
            rcs=rcs_arr,
            rcs_domain="complex_amplitude",
            source_path=str(path),
            history=history,
            units={"azimuth": "deg", "elevation": "deg", "frequency": "GHz"},
        )
