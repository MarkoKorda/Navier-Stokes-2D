import numpy as np 
import matplotlib.pyplot as plt
import cmasher as cmr 
from tqdm import tqdm

#constants
N_POINTS_Y = 21
ASPECT_RATIO = 10
KINEMATIC_VISCOSITY = 0.02
TIME_STEP_LENGTH = 0.001
N_TIME_STEPS = 5000
PLOT_EVERY = 50

N_PRESSURE_POISSON_ITERATIONS = 50

def main():
    cell_length = 1.0 / (N_POINTS_Y - 1)

    n_points_x = (N_POINTS_Y - 1) * ASPECT_RATIO + 1

    x = np.linspace(0.0, 1.0 * ASPECT_RATIO, n_points_x)
    y = np.linspace(0.0, 1.0, N_POINTS_Y)

    X, Y = np.meshgrid(x, y)

    #initial conditions
    u_prev = np.ones((N_POINTS_Y + 1, n_points_x))
    # boundary conditions
    u_prev[0, :] = - u_prev[1, :]
    u_prev[-1, :] = - u_prev[-2, :]

    v_prev = np.zeros((N_POINTS_Y, n_points_x + 1))

    p_prev = np.zeros((N_POINTS_Y + 1, n_points_x + 1))


    u_tent = np.zeros_like(u_prev)
    u_next = np.zeros_like(u_prev)

    v_tent = np.zeros_like(v_prev)
    v_next = np.zeros_like(v_prev)

    plt.rcParams.update({"font.size": 20})
    plt.figure(figsize=(15, 10))

    for iter in tqdm(range(N_TIME_STEPS)):
        diffusion_x = KINEMATIC_VISCOSITY * (
            (
                u_prev[1:-1, 2:  ]
                +
                u_prev[2:  , 1:-1]
                + 
                u_prev[1:-1, 0:-2]
                +
                u_prev[0:-2, 1:-1]
                - 4 * 
                u_prev[1:-1, 1:-1]
            ) / (
                cell_length**2
            )
        )

        convection_x = (
            (
                u_prev[1:-1, 2:  ]**2
                -
                u_prev[1:-1, 0:-2]**2
            ) / (
                2 * cell_length
            )
            +
            (
                v_prev[1:  , 1:-2]
                +
                v_prev[1:  , 2:-1]
                +
                v_prev[ :-1, 1:-2]
                +
                v_prev[ :-1, 2:-1]
            ) / 4
            *
            (
                u_prev[2:  , 1:-1]
                -
                u_prev[0:-2, 1:-1]
            ) / (
                2 * cell_length
            )
        )

        pressure_gradient_x = (
            (
                p_prev[1:-1, 2:-1]
                -
                p_prev[1:-1, 1:-2]
            ) / (
                cell_length
            )
        )

        u_tent[1:-1, 1:-1] = (
            u_prev[1:-1, 1:-1]
            +
            TIME_STEP_LENGTH
            *
            (
                -
                pressure_gradient_x
                +
                diffusion_x
                -
                convection_x
            )
        )

        #boundary conditions
        u_tent[1:-1, 0] = 1.0
        u_tent[1:-1, -1] = u_tent[1:-1, -2]
        u_tent[0, :] = - u_tent[1, :]
        u_tent[-1, :] = - u_tent[-2, :]


        diffusion_y = KINEMATIC_VISCOSITY * (
            (
                v_prev[1:-1, 2:  ]
                +
                v_prev[2:  , 1:-1]
                +
                v_prev[1:-1, 0:-2]
                +
                v_prev[0:-2, 1:-1]
                - 
                4 * v_prev[1:-1, 1:-1]
            ) / (
                cell_length**2
            )
        )

        convection_y = (
            (
                u_prev[2:-1, 1:  ]
                +
                u_prev[2:-1,  :-1]
                + 
                u_prev[1:-2, 1:  ]
                +
                u_prev[1:-2,  :-1]
            ) / 4
            *
            (
                v_prev[1:-1, 2:  ]
                -
                v_prev[1:-1, 0:-2]
            ) / (
                2 * cell_length
            )
            +
            (
                v_prev[2: , 1:-1]**2
                -
                v_prev[0:-2, 1:-1]**2
            ) / (
                2 * cell_length
            )
        )

        pressure_gradient_y = (
            (
                p_prev[2:-1, 1:-1]
                -
                p_prev[1:-2, 1:-1]
            ) / (
                cell_length
            )
        )

        v_tent[1:-1, 1:-1] = (
            v_prev[1:-1, 1:-1]
            +
            TIME_STEP_LENGTH
            *
            (
                -
                pressure_gradient_y
                + 
                diffusion_y
                -
                convection_y
            )
        )

        #boundary conditions
        v_tent[1:-1, 0] = - v_tent[1:-1, 1]
        v_tent[1:-1, -1] = v_tent[1:-1, -2]
        v_tent[0, :] = 0.0
        v_tent[-1, :] = 0.0

        divergence = (
            (
                u_tent[1:-1, 1:  ]
                -
                u_tent[1:-1,  :-1]
            ) / (
                cell_length
            )
            +
            (
                v_tent[1:  , 1:-1]
                -
                v_tent[ :-1, 1:-1]
            ) / (
                cell_length
            )
        )

        p_poisson_rhs = divergence / TIME_STEP_LENGTH

        p_correction_prev = np.zeros_like(p_prev)
        for _ in range(N_PRESSURE_POISSON_ITERATIONS):
            p_correction_next = np.zeros_like(p_correction_prev)
            p_correction_next[1:-1, 1:-1] = 1/4 * (
                p_correction_prev[1:-1, 2:  ]
                +
                p_correction_prev[2:  , 1:-1]
                +
                p_correction_prev[1:-1,  :-2]
                +
                p_correction_prev[ :-2, 1:-1]
                -
                cell_length**2
                *
                p_poisson_rhs
            )

            #boundary conditions
            p_correction_next[1:-1, 0] = p_correction_next[1:-1, 1]
            p_correction_next[1:-1, -1] = - p_correction_next[1:-1, -2]
            p_correction_next[0, :] = p_correction_next[1, :]
            p_correction_next[-1, :] = p_correction_next[-2, :]

            p_correction_prev = p_correction_next

        p_next = p_prev + p_correction_next

        p_correction_gradient_x = (
            (
                p_correction_next[1:-1, 2:-1]
                -
                p_correction_next[1:-1, 1:-2]
            ) / (
                cell_length
            )
        )

        u_next[1:-1, 1:-1] = (
            u_tent[1:-1, 1:-1]
            -
            TIME_STEP_LENGTH
            *
            p_correction_gradient_x
        )

        p_correction_gradient_y = (
            (
                p_correction_next[2:-1, 1:-1]
                -
                p_correction_next[1:-2, 1:-1]
            ) / (
                cell_length
            )
        )

        v_next[1:-1, 1:-1] = (
            v_tent[1:-1, 1:-1]
            -
            TIME_STEP_LENGTH
            *
            p_correction_gradient_y
        )

        #boundary conditions
        u_next[1:-1, 0] = 1.0
        inflow_mass_rate_next = np.sum(u_next[1:-1, 0])
        outflow_mass_rate_next = np.sum(u_next[1:-1, -2])
        u_next[1:-1, -1] = u_next[1:-1, -2] * inflow_mass_rate_next / outflow_mass_rate_next
        u_next[0, :] = - u_next[1, :]
        u_next[-1, :] = - u_next[-2, :]

        v_next[1:-1, 0] = - v_next[1:-1, 1]
        v_next[1:-1, -1] = v_next[1:-1, -2]
        v_next[0, :] = 0.0
        v_next[-1, :] = 0.0

        u_prev = u_next
        v_prev = v_next
        p_prev = p_next

        """
        inflow_mass_rate_next = np.sum(u_next[1:-1, 0])
        outflow_mass_rate_next = np.sum(u_next[1:-1, -1])
        print(f"Inflow: {inflow_mass_rate_next}")
        print(f"Outflow: {outflow_mass_rate_next}")
        print()
        """

        #visualisation
        if iter % PLOT_EVERY == 0:
            u_centered = (
                (
                    u_next[1:  , :]
                    +
                    u_next[ :-1, :]
                ) / 2
            )
            v_centered = (
                (
                    v_next[:, 1:  ]
                    +
                    v_next[:,  :-1]
                ) / 2
            )
            p_centered = (
                (
                    p_next[1:  , 1: ]
                    +
                    p_next[ :-1, 1:]
                    +
                    p_next[1:  , :-1]
                    +
                    p_next[ :-1, :-1]
                ) / 4
            )

            plt.contourf(
                X,
                Y,
                np.sqrt(u_centered**2 + v_centered**2),
                levels=20,
                cmap=cmr.ember,
                vmin=-0.5,
                vmax=1.5,
            )

            """
            plt.contourf(
                X, 
                Y, 
                p_centered, 
                levels=100,
            )
            """

            cbar = plt.colorbar(label=r"$|u|$ [m/s]", orientation="horizontal")

            plt.quiver(
                X[:, ::10],
                Y[:, ::10],
                u_centered[:, ::10],
                v_centered[:, ::10],
                alpha=0.6,
            )

            plt.plot(
                5 * cell_length + u_centered[:, 5],
                Y[:, 5],
                color="black",
                linewidth=3,
            )

            plt.plot(
                50 * cell_length + u_centered[:, 50],
                Y[:, 50],
                color="black",
                linewidth=3,
            )

            plt.plot(
                150 * cell_length + u_centered[:, 150],
                Y[:, 150],
                color="black",
                linewidth=3,
            )

            plt.xlabel(r"$x$ [m]", labelpad=-10)
            plt.ylabel(r"$y$ [m]")
            plt.draw()
            plt.pause(0.05)
            plt.clf()

    plt.contourf(
        X,
        Y,
        np.sqrt(u_centered**2 + v_centered**2),
        levels=20,
        cmap=cmr.ember,
        vmin=-0.5,
        vmax=1.5,
    )

    cbar = plt.colorbar(label=r"$|u|$ [m/s]", orientation="horizontal")

    """
    plt.streamplot(
        X,
        Y,
        u_centered,
        v_centered,
        color="black",
    )
    """
    
    plt.quiver(
        X[:, ::10],
        Y[:, ::10],
        u_centered[:, ::10],
        v_centered[:, ::10],
        alpha=0.6,
    )

    plt.plot(
        5 * cell_length + u_centered[:, 5],
        Y[:, 5],
        color="black",
        linewidth=3,
    )

    plt.plot(
        50 * cell_length + u_centered[:, 50],
        Y[:, 50],
        color="black",
        linewidth=3,
    )

    plt.plot(
        150 * cell_length + u_centered[:, 150],
        Y[:, 150],
        color="black",
        linewidth=3,
    )

    plt.xlabel(r"$x$ [m]", labelpad=-10)
    plt.ylabel(r"$y$ [m]")
    plt.show()

    #usporedba s analitickim
    dpdx = np.average((p_centered[1:-1, -1] - p_centered[1:-1, -2]) / cell_length)
    print(dpdx)

    x_numeric = y
    y_numeric = u_centered[:, -1]

    x_analitic = np.linspace(0.0, 1.0, 100)
    y_analitic = dpdx / (2 * KINEMATIC_VISCOSITY) * x_analitic * (x_analitic - 1.0)

    plt.figure(figsize=(8, 6))
    plt.rcParams.update({"font.size": 12})
    plt.plot(x_analitic, y_analitic, color="blue", label="Analitičko rješenje")
    plt.scatter(x_numeric, y_numeric, color="red", marker="+", label="Numeričko rješenje")
    plt.xlabel(r"$y$ [m]")
    plt.ylabel(r"$u$ [m/s]")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
