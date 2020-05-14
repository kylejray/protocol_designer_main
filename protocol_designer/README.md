### protocol_designer

This package is designed to be used with Greg's infoenginessims package, which simulates the dynamics of the System class.

The protocol_designer package works off of three main classes:

    1 the Protocol class, stores a time dependent signal: N_p parameters as a function of time
    2 the Potential class, stores a position dependent potential in N_d dimensions with N_p variable parameters
    3 the System class, applies the time dependent signal to the potential-- creating a time dependent potential

    The system class also has methods that
        * generate equilibrium distributions associated with the potential at arbitrary times
        * generate visualizations of the time dependent potential
        * allow interface with infoenginessims by having a standard agreed upon api
    
    Among these api conventions are...
    
    The format of our coordiante arrays, coords: an ndarray of shape [N_coords, N_dim, 2]:
        * N_coords is the number of set of coordiantes
        * N_dim is the dinensionality of our system
        * The size 2 dimension holds the position(index 0) and velocity(index 1) corrdinate for each dimension and coordinate
    
    The methods of System that return the energy/potential/conservative_force of a particular point in phase space at a time, t:
        * System.get_potential(coords, t)
            returns an ndarray of length (N_coords)
        * System.get_potential(coords, t)
            returns an ndarray of length (N_coords)
        * System.get_external_force(coords, t)
            returns an ndarray of shape (N_coords, N_dim)

    

    

