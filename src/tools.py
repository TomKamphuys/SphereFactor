import glob
import os.path

import soundfile as sf
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import spherical_yn, spherical_jn, sph_harm_y


def spherical_hn1(n, z):
    """ Spherical Hankel functions of the first kind. """
    return spherical_jn(n, z) + 1j * spherical_yn(n, z)


def to_pressure(mag: np.ndarray) -> np.ndarray:
    """Convert magnitude to pressure."""
    return (10 ** (mag / 20)) * 20e-6


def read_nfs_measurements(subdir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read near-field scanning measurements from WAV files.

    Args:
        subdir: Subdirectory name containing the measurements

    Returns:
        tuple: (p, r, theta, z, f) containing pressure values, radius, angles,
               z-coordinates, and frequencies
    """
    directory = os.path.join('Measurements', subdir)
    files = glob.glob(os.path.join(directory, '*.wav'))
    # files = glob.glob(f'{directory}/*.wav')


    p = []
    r = []
    theta = []
    z = []
    freq = []

    for file in files:
        filename, freq, pt = process_file(file)

        p.append(pt)
        r_val, theta_val, z_val = extract_measurement_position_from_filename(filename)
        r.append(r_val)
        theta.append(theta_val)
        z.append(z_val)

    p = np.array(p)
    r = np.array(r)
    theta = np.array(theta)
    z = np.array(z)
    f = freq

    return p, r, theta, z, f


def process_file(file: str) -> tuple[str, np.ndarray, np.ndarray]:
    filename = os.path.splitext(os.path.basename(file))[0]
    # Read the audio file
    hh, fs = sf.read(file)
    freq, p, = ir_to_fr(fs, hh)  # The same up to here with mataa. Now some magic is being done

    p_ref = 1.0
    p_rms = abs(p)
    mag = 20*np.log10(p_rms/p_ref)
    phase = np.unwrap(np.angle(p))
    p = to_pressure(mag) * np.exp(1j*phase)

    return filename, freq, p


def ir_to_fr(fs: float, hh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts impulse response to frequency response.

    This function computes the frequency response of a system given
    its impulse response (`hh`) and sampling frequency (`fs`). It
    calculates the frequency bins, magnitudes, and phases by performing
    a Fast Fourier Transform (FFT) on the impulse response.

    :param fs: Sampling frequency of the input signal.
        This defines how frequently the signal samples were taken.
        Must be provided as a floating-point value.
    :param hh: Impulse response of the system.
        A sequence of data points representing the response of the
        system over time, provided as an array-like input.
    :return: A tuple containing three elements:
        - freq: Array of frequency bins corresponding to the FFT.
        - mag: Array of magnitudes computed from the FFT of `hh`.
        - phase: Array of phases computed from the FFT of `hh`.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    # For now using placeholder frequency response calculation
    n_fft = len(hh)
    freq = np.fft.rfftfreq(n_fft, 1 / fs)[1:]  # TODO this could be done only once
    p = np.fft.rfft(hh, n_fft)[1:]  # TODO It looks like the imaginary part is multiplied by -1 wrt octave
    return freq, p


def cart2sph_phys(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates to spherical coordinates (physics convention).

    Args:
        x: x coordinates
        y: y coordinates
        z: z coordinates

    Returns:
        tuple: (phi, theta, r) containing the azimuthal angle, polar angle, and radius
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)
    return phi, theta, r


def correct_for_setup(theta: np.ndarray, r: np.ndarray, beam_offset: float, arm_offset: float,
                      arm_angle: float) -> np.ndarray:
    """Correct angle based on setup parameters.

    Args:
        theta: angle to correct
        r: radius values
        beam_offset: beam offset in meters
        arm_offset: arm offset in meters
        arm_angle: arm angle in degrees

    Returns:
        numpy.ndarray: Corrected angle values
    """
    delta = beam_offset - arm_offset + (0.8 - r) * arm_angle / 180 * np.pi
    delta_angle = np.arctan2(delta, r)
    return theta + delta_angle


def take_virtual_measurement(cd: np.ndarray, phi: np.ndarray, theta: np.ndarray, r: np.ndarray, frequencies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Take a virtual measurement using the coefficients and frequencies.
    """
    n = calc_n_max(cd.shape[0])

    temp = 273.15 + 20
    kr = calc_kr(r, frequencies, temp)

    angular_part = calc_angular_part(phi, theta, n)
    spl = np.zeros((len(phi), len(kr)))
    phase = np.zeros((len(phi), len(kr)))

    for ind, k in enumerate(kr):
        sph_hn1 = calc_radial_part(np.array(k), n) * angular_part
        p_recon = sph_hn1 @ cd[:, ind]
        spl[:, ind] = db_spl(p_recon)
        phase[:, ind] = np.angle(p_recon)/np.pi*180

    return spl, phase


def vertical_directivity(cd: np.ndarray, frequencies: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the vertical directivity pattern.

    Args:
        cd: Coefficient array
        frequencies: Frequencies array

    Returns:
        tuple: (out_recon, angle, frequencies) containing reconstructed output,
               angles, and frequencies
    """
    distance = np.array([10])  # m

    num_points = 100
    angle = np.linspace(-np.pi, np.pi, num_points)
    x_recon = distance * np.cos(angle)
    y_recon = np.zeros_like(x_recon)
    z_recon = distance * np.sin(angle)

    phi, theta, r = cart2sph_phys(x_recon, y_recon, z_recon)

    spl = take_virtual_measurement(cd, phi, theta, distance, frequencies)[0]

    return spl, angle, frequencies


def convert_to_coefficients(z: np.ndarray, r: np.ndarray, theta: np.ndarray, p: np.ndarray, f: np.ndarray, n: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """Convert measurements to coefficients.

    Args:
        z: Z coordinates
        r: Radius values
        theta: Theta angles
        p: Pressure values
        f: Frequencies

    Returns:
        tuple: (CD, fit_error, CD_tot) containing coefficients and error metrics
    """
    temp = 273.15 + 20

    beam_offset = 0.025  # meter
    arm_offset = 0.0023  # meter
    arm_angle = 1.46  # degrees

    theta = correct_for_setup(theta, r, beam_offset, arm_offset, arm_angle)

    #  cyl2cart
    x, y = r * np.cos(theta), r * np.sin(theta)

    #  acoustic center correction (no correction at the moment)
    x_meas, y_meas, z_meas = x, y, z

    phi, theta, r = cart2sph_phys(x_meas.T, y_meas.T, z_meas.T)

    sph_harm_vals = calc_angular_part(phi, theta, n)

    cd = np.zeros((((n + 1) ** 2), len(f)), dtype=complex)
    cd_tot = np.zeros((2 * ((n + 1) ** 2), len(f)), dtype=complex)
    fit_error = np.zeros(len(f))

    for ind, freq in enumerate(f):
        kr = calc_kr(r, np.array(freq), temp)
        p_meas = p[:, ind]

        outgoing = calc_radial_part(kr, n) * sph_harm_vals
        incoming = calc_radial_part_in(kr, n) * sph_harm_vals

        total = np.hstack((outgoing, incoming))

        cd_vec = np.linalg.lstsq(total, p_meas, rcond=None)[0]

        fit_error[ind] = calc_error(p_meas, total, cd_vec)[0]

        cd[:, ind] = get_outgoing_coefficients(cd_vec)
        cd_tot[:, ind] = cd_vec

    return cd, fit_error, cd_tot


def calc_n_max(nr: int) -> int:
    """Calculate n as square root of nr minus 1.

    Args:
        nr: Input number

    Returns:
        float: Square root of nr minus 1
    """
    return int(np.sqrt(nr)) - 1


def spherical_harmonic(n: int, m: int, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Calculate spherical harmonics using scipy.special.sph_harm.

    TODO: look at https://github.com/NVIDIA/torch-harmonics sometime

    """
    y = sph_harm_y(n, abs(m), theta, phi)
    # Linear combination of Y_l,m and Y_l,-m to create the real form.
    if m < 0:
        y = y.imag  # np.sqrt(2) * (-1)**m * : TODO this scaling is a difference with octave
    elif m > 0:
        y = y.real

    return y


def calc_angular_part(phi: np.ndarray, theta: np.ndarray, n: int) -> np.ndarray:
    """Calculate angular part using spherical harmonics.

    Args:
        phi: azimuthal angles in radians
        theta: polar angles in radians
        n: maximum degree for spherical harmonics

    Phi and theta are expected to have the same shape.

    Returns:
        numpy.ndarray: Array of shape (len(phi), (N+1)^2) containing spherical harmonics
    """
    out = np.zeros((phi.size, (n + 1) ** 2), dtype=complex)

    for n in range(n + 1):
        for m in range(-n, n + 1):
            j = n ** 2 + n + m
            # Note: scipy.special.sph_harm expects opposite order of angles
            # compared to typical physics convention
            out[:, j] = spherical_harmonic(n, m, phi, theta)

    return out


def calc_error(p_meas: np.ndarray, psi: np.ndarray, cd_vec: np.ndarray) -> tuple[float, float]:
    """Calculate error metrics between measured and modeled pressure values.

    Args:
        p_meas: Measured pressure values
        psi: Transfer matrix
        cd_vec: Coefficient vector

    Returns:
        tuple: (error_decibel, error_percentage) containing error in dB and percentage
    """
    p_mod = psi @ cd_vec
    error_percentage = np.sum(np.abs(p_mod - p_meas) ** 2) / np.sum(np.abs(p_meas) ** 2) * 100
    error_decibel = 10 * np.log10(error_percentage / 100)
    return error_decibel, error_percentage


def calc_radial_part(kr: np.ndarray, n: int) -> np.ndarray:
    """Calculate radial part using spherical Hankel functions.

    Args:
        kr: Wave numbers array
        n: Maximum degree for spherical harmonics

    Returns:
        numpy.ndarray: Array of shape (len(kr), (n+1)^2) containing radial values
    """
    out = np.zeros((kr.size, (n + 1) ** 2), dtype=complex)

    for n_val in range(n + 1):
        hn1 = spherical_hn1(n_val, kr)
        for m in range(-n_val, n_val + 1):
            j = n_val ** 2 + n_val + m
            out[:, j] = hn1

    return out


def calc_radial_part_in(kr: np.ndarray, n: int) -> np.ndarray:
    """Calculate radial part using spherical Bessel functions.

    Args:
        kr: Wave numbers array
        n: Maximum degree for spherical harmonics

    Returns:
        numpy.ndarray: Array of shape (len(kr), (N+1)^2) containing radial values
    """
    out = np.zeros((kr.size, (n + 1) ** 2), dtype=complex)

    for n_val in range(n + 1):
        jn = spherical_jn(n_val, kr)
        for m in range(-n_val, n_val + 1):
            j = n_val ** 2 + n_val + m
            out[:, j] = jn

    return out


def calc_kr(r: np.ndarray, frequencies: np.ndarray, temp: float) -> np.ndarray:
    """Calculate wave numbers based on radius, frequencies, and temperature.

    Args:
        r: something
        frequencies: Frequencies array
        temp: Temperature value

    Returns:
        numpy.ndarray: Array of calculated wave numbers
    """
    r_air = 287.058
    c = np.sqrt(1.4 * r_air * temp)
    omega = 2 * np.pi * frequencies
    kr = r * omega / c
    return kr


def get_outgoing_coefficients(cd: np.ndarray) -> np.ndarray:
    """Extract outgoing coefficients from the CD array.

    Args:
        cd: Input coefficient array

    Returns:
        numpy.ndarray: First half of rows from the input array
    """
    n = cd.shape[0]
    return cd[:n // 2]


def db_spl(x: np.ndarray) -> np.ndarray:
    """Convert sound pressure to dB SPL.

    Args:
        x: Sound pressure values (can be complex)

    Returns:
        numpy.ndarray: Sound pressure level in dB
    """
    return 20 * np.log10(np.abs(x) / 20e-6)


def extract_measurement_position_from_filename(filename: str) -> tuple[float, float, float]:
    """Extract measurement position coordinates from the file with `filename` as its name.

    Args:
        filename: String containing position coordinates in format '(r,theta,z)'

    Returns:
        tuple: (r, theta, z) containing radius in meters, angle in radians, and z-position in meters
    """
    position = filename.replace('(', '').replace(')', '')
    position = position.split(',')

    r = float(position[0]) / 1000  # convert to meters
    theta = float(position[1]) * np.pi / 180  # convert to radians
    z = float(position[2]) / 1000  # convert to meters

    return r, theta, z


def sph2cart_phys(phi, theta, r):
    """Convert spherical to Cartesian coordinates (physics convention)"""
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z


def view_modes(N):
    nr = 100

    # Create meshgrid
    theta, phi = np.meshgrid(np.linspace(0, np.pi, nr), np.linspace(-np.pi, np.pi, nr))

    # Flatten arrays
    theta = theta.flatten()
    phi = phi.flatten()

    # Create 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot for each n and m
    for n in range(N + 1):
        for m in range(-n, n + 1):
            value = spherical_harmonic(n, m, phi, theta)
            teken = np.sign(value)
            r = np.abs(value)

            # Convert to Cartesian coordinates
            x, y, z = sph2cart_phys(phi, theta, r)

            # Reshape for surface plot
            x = x.reshape(nr, nr)
            y = y.reshape(nr, nr) + m  # Offset in y direction
            z = z.reshape(nr, nr) - 2 * n  # Offset in z direction
            teken = teken.reshape(nr, nr)

            # Create surface plot
            surf = ax.plot_surface(x, y, z, facecolors=plt.cm.summer(teken))

    # Set plot properties
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    plt.colormap = 'summer'
    plt.colorbar(surf)
    ax.axis('off')

    # Add lighting
    ax.azim = -60
    ax.elev = 20
    ax.view_init(0, 0, 0)

    plt.show()
