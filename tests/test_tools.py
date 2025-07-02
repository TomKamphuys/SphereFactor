import matplotlib.pyplot as plt
import numpy as np
import os
import pytest
from src.tools import extract_measurement_position_from_filename, spherical_hn1, read_nfs_measurements, \
    convert_to_coefficients, calc_radial_part, calc_angular_part, spherical_harmonic, cart2sph_phys, \
    vertical_directivity, process_file


def test_calc_radial_part_with_multiple_kr():
    kr = np.array([1.0, 2.0, 3.0])
    n = 1
    result = calc_radial_part(kr, n)
    expected_shape = (3, 4)  # (len(kr), (n+1)^2)
    assert result.shape == expected_shape
    assert np.iscomplexobj(result)  # Outputs should be complex numbers


def test_calc_radial_part_with_high_degree():
    kr = np.array([1])
    n = 0
    result = calc_radial_part(kr, n)
    expected = 0.841470984807896 - 1j * 0.540302305868140
    expected_shape = (1, 1)
    assert result == pytest.approx(expected)
    assert result.shape == expected_shape
    assert np.iscomplexobj(result)  # Outputs should be complex numbers


def test_calc_angular_part_with_zeros():
    phi = np.array([0])
    theta = np.array([0])
    n = 0
    result = calc_angular_part(phi, theta, n)
    expected = 0.282094791773878
    expected_shape = (1, 1)
    assert result == pytest.approx(expected)
    assert result.shape == expected_shape


def test_cart2sph_phys_z():
    x = np.array([0])
    y = np.array([0])
    z = np.array([1])

    phi, theta, r = cart2sph_phys(x, y, z)
    expected_theta = np.array([0.0])
    expected_r = np.array([1.0])
    assert theta == pytest.approx(expected_theta)
    assert r == pytest.approx(expected_r)


def test_cart2sph_phys_x():
    x = np.array([1])
    y = np.array([0])
    z = np.array([0])

    phi, theta, r = cart2sph_phys(x, y, z)
    expected_phi = np.array([0.0])
    expected_theta = np.array([np.pi/2])
    expected_r = np.array([1.0])
    assert phi == pytest.approx(expected_phi)
    assert theta == pytest.approx(expected_theta)
    assert r == pytest.approx(expected_r)


def test_cart2sph_phys_y():
    x = np.array([0])
    y = np.array([1])
    z = np.array([0])

    phi, theta, r = cart2sph_phys(x, y, z)
    expected_phi = np.array([np.pi/2])
    expected_theta = np.array([np.pi / 2])
    expected_r = np.array([1.0])
    assert phi == pytest.approx(expected_phi)
    assert theta == pytest.approx(expected_theta)
    assert r == pytest.approx(expected_r)


def test_spherical_harmonic_1():
    phi = np.array([0.1])
    theta = np.array([1])
    n = 1
    m = -1
    result = spherical_harmonic(n, m, phi, theta)
    expected = np.array([-2.902390055757528e-2])
    assert result == pytest.approx(expected)


def test_spherical_harmonic_2():
    phi = np.array([0.1])
    theta = np.array([1])
    n = 1
    m = 1
    result = spherical_harmonic(n, m, phi, theta)
    expected = np.array([-0.289270896633388])
    assert result == pytest.approx(expected)


def test_spherical_harmonic_3():
    phi = np.array([0.0])
    theta = np.array([0.0])
    n = 0
    m = 0
    result = spherical_harmonic(n, m, phi, theta)
    expected = np.array([0.282094791773878])
    assert result == pytest.approx(expected)


def test_spherical_harmonic_4():
    phi = np.array([1.0])
    theta = np.array([0.0])
    n = 0
    m = 0
    result = spherical_harmonic(n, m, phi, theta)
    expected = np.array([0.282094791773878])
    assert result == pytest.approx(expected)


def test_spherical_harmonic_5():
    phi = np.array([0.0])
    theta = np.array([0.0])
    n = 1
    m = 0
    result = spherical_harmonic(n, m, phi, theta)
    expected = np.array([0.488602511902920])
    assert result == pytest.approx(expected)


def test_spherical_harmonic_6():
    phi = np.array([0.0])
    theta = np.array([0.0])
    n = 1
    m = -1
    result = spherical_harmonic(n, m, phi, theta)
    expected = np.array([0])
    assert result == pytest.approx(expected)


def test_spherical_harmonic_7():
    phi = np.array([0.0])
    theta = np.array([0.0])
    n = 1
    m = 1
    result = spherical_harmonic(n, m, phi, theta)
    expected = np.array([0])
    assert result == pytest.approx(expected)


def test_calc_angular_part_with_some_pseudo_random_values():
    phi = np.array([0.1])
    theta = np.array([1])
    n = 1
    result = calc_angular_part(phi, theta, n)
    expected = np.array([2.820947917738782e-1, -2.902390055757528e-2, 2.6399306383411219e-1, -2.89270896633383e-1])
    expected_shape = (1, 1)
    assert result == pytest.approx(expected)
    assert result.shape == expected_shape


def test_calc_radial_part_with_basic_inputs():
    kr = np.array([1.0])
    n = 0
    result = calc_radial_part(kr, n)
    expected_shape = (1, 1)  # (len(kr), (n+1)^2)
    assert result.shape == expected_shape
    assert np.iscomplexobj(result)  # Outputs should be complex numbers


def test_process_file_valid_input():
    file = "../Measurements/11062025/(48.975, 89.998, 288.212).wav"
    filename, freq, pt = process_file(file)
    assert freq[0] == np.array([9.997732940376331])[0]
    assert pt[0] == pytest.approx(np.array([-1.028980151490744e-7 - 1j * 1.830585819168071e-9])[0])
    assert filename == "(48.975, 89.998, 288.212)"


def test_extract_measurement_position_valid_input():
    filename = "(1500.000,90.000,2000.000)"
    result = extract_measurement_position_from_filename(filename)
    expected = (1.5, 1.5707963267948966, 2.0)  # 1.5 m, 90° in radians (π/2), 2.0 m
    assert result == pytest.approx(expected)


def test_extract_measurement_position_negative_values():
    filename = "(-1000.000,180.000,-3000.000)"
    result = extract_measurement_position_from_filename(filename)
    expected = (-1.0, 3.141592653589793, -3.0)  # -1.0 m, 180° in radians (π), -3.0 m
    assert result == pytest.approx(expected)


def test_extract_measurement_position_decimal_values():
    filename = "(500.500,45.500,1500.750)"
    result = extract_measurement_position_from_filename(filename)
    expected = (0.5005, 0.7941248096574199, 1.50075)  # 0.5005 m, 45.5° in radians, 1.50075 m
    assert result == pytest.approx(expected)


def test_extract_measurement_position_zero_values():
    filename = "(0,0,0)"
    result = extract_measurement_position_from_filename(filename)
    expected = (0.0, 0.0, 0.0)  # all values are zero
    assert result == pytest.approx(expected)


def test_extract_measurement_position_large_values():
    filename = "(100000,360,500000)"
    result = extract_measurement_position_from_filename(filename)
    expected = (100.0, 6.283185307179586, 500.0)  # 100 m, 360° in radians (2π), 500 m
    assert result == pytest.approx(expected)


def test_spherical_hn1_basic_case():
    n, z = 0, 1.0
    result = spherical_hn1(n, z)
    expected = 0.841470984807896 - 1j * 0.540302305868140
    assert result == pytest.approx(expected, rel=1e-9)


def test_spherical_hn1_large_inputs():
    n, z = 10, 100.0
    result = spherical_hn1(n, z)
    expected = -1.956578597134297e-4 + 1j * 1.002577737363615e-2
    assert result == pytest.approx(expected, rel=1e-9)


def test_read_nfs_measurements():
    os.chdir('..')
    p, r, theta, z, f = read_nfs_measurements('11062025')
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(f, abs(p[0, :]))  # Plot first measurement position
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Pressure Magnitude (Pa)')
    plt.title('Pressure vs Frequency')
    plt.grid(True)
    plt.savefig('pressure_vs_frequency.png')
    plt.close()

    assert p.shape == (991, 2205)


def test_read_and_convert():
    os.chdir('..')
    p, r, theta, z, f = read_nfs_measurements('11062025')

    cd, fit_error, cd_tot = convert_to_coefficients(z, r, theta, p, f, 8)

    plt.figure(figsize=(10, 6))
    plt.semilogx(f, fit_error)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Fit error [dB]')
    plt.title('Fit error vs Frequency')
    plt.grid(True)
    plt.savefig('fit_error_vs_frequency.png')
    plt.close()

    spl, angle, frequencies = vertical_directivity(cd, f)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(frequencies, angle, spl, shading='auto')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Vertical directivity angle [deg]')
    plt.title('Vertical directivity angle vs Frequency')
    plt.savefig('vertical_directivity.png')
    plt.close()