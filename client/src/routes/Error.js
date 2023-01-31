import "../styles/Error.css"

function Error() {
    return(
        <div className="error-block d-flex flex-column">
            <div className="error-block__header">
                <span>Error 404:</span>
            </div>
            <div className="error-block__text">
                <span>Not found</span>
            </div>
        </div>
    )
}

export default Error