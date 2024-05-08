//NOTE: THIS A TEMPORARY JS FILE FOR EXPERIMENTATION

document.addEventListener("DOMContentLoaded", function () {
    const textarea = document.querySelector(".message_input");
    textarea.addEventListener("input", () => {
        textarea.style.height = "auto";
        textarea.style.height = textarea.scrollHeight + "px";
    });
});


function fetchAndDisplayGraph($message) {
    // Fetch graph data from the server
    fetch('/get_graph')
        .then(response => response.json())
        .then(data => {
            var [remaining_text, graphData] = data; // Destructure the response tuple

            // If graphData is not null, render the Plotly graph
            if (graphData !== null) {
                var graphContainer = $('<div id="graph"></div>'); // Create a container for the Plotly graph
                // Clear the container before rendering the new plot
                Plotly.newPlot(graphContainer[0], [graphData.data], graphData.layout ); //graphData.layout  Render the Plotly graph

                // Append the graph container inside the bot message box
                $message.find('.text_wrapper').append(graphContainer);
            }
        })

        .catch(error => console.log('Warning fetching graph data:', error));
}

(function () {
        // Define a JavaScript function for creating a message
        var Message;
        Message = function (arg) {
            this.text = arg.text, this.message_side = arg.message_side, this.buttons = arg.buttons, this.graphData = arg.graphData;
            this.draw = function (_this) {
                return function () {
                    var $message;
                    $message = $($('.message_template').clone().html());
                    $message.addClass(_this.message_side).find('.text').html(_this.text);

                    // Add buttons
                    var $buttonsContainer = $message.find('.text_wrapper');
                    _this.buttons.forEach(function (button) {
                        var $button = $('<button class="message_button">' + button.text + '</button>');
                        $button.click(function () {
                            // Handle button click event here
                            $('.message_input').val(button.text)
                        });
                        $buttonsContainer.append($button);
                    });

                    // Add copy icon to bot messages
                    if (_this.message_side === 'left') {
                        var $copyIcon = $('<img class="copy-icon" src="../static/copy_icon.png" alt="Copy message" title="Copy message">');
                        $copyIcon.css({
                            'width': '20px',   // Adjust the width of the icon
                            'height': '20px',  // Adjust the height of the icon
                            'position': 'absolute',
                            'bottom': '10px',
                            'left': '5px'
                        });
                        $copyIcon.click(function () {
                            var textToCopy = $message.find('.text').text();
                            navigator.clipboard.writeText(textToCopy)
                                .then(function () {
                                    console.log('Text copied to clipboard');
                                })
                                .catch(function (err) {
                                    console.error('Unable to copy text to clipboard: ', err);
                                });
                        });
                        $copyIcon.css('cursor', 'pointer'); // Set cursor to pointer
                        $copyIcon.hover(function () {
                            $(this).css('cursor', 'pointer'); // Change cursor to pointer on hover
                        }, function () {
                            $(this).css('cursor', 'default'); // Revert cursor to default when not hovered
                        });
                        $message.find('.text').append($copyIcon);
                    }

                    //This part renders the graph
                    // if (_this.message_side === 'left' && !_this.text.includes('Hello! I\'m CodeGenie')) {
                    //     console.log('appending image in left message');
                    //     fetchAndDisplayGraph($message);
                    // }

                    $('.messages').append($message);
                    return setTimeout(function () {
                        return $message.addClass('appeared');
                    }, 0);
                };
            }(this);
            return this;
        };

        // Execute when the document is ready
        $(function () {

                // Function to get the text of the message
                function getMessageText() {
                    var $message_input;
                    $message_input = $('.message_input');
                    return $message_input.val();
                }

                // Function to send a message
                function sendMessage(text) {
                    var $messages, message;
                    if (text.trim() === '') {
                        return;
                    }
                    $('.message_input').val('');
                    $('.message_input').css('height', 'auto');

                    $messages = $('.messages');

                    // Set message_side based on whether the message is from the user or chatbot
                    var message_side = 'right';

                    message = new Message({
                        text: text,
                        message_side: message_side,
                        buttons: [] // No buttons for user message

                    });

                    // Draw user message
                    message.draw();

                    // Get the selected option value
                    var ChatMode = $('#chatType').val();

                    // Call getResponse() to get the chatbot's response
                    $.get("/get", {msg: text, option: ChatMode}).done(function (data) {
                        // Extract remaining text from the response tuple

                        // if (Array.isArray(data)) {
                        //     // Extract remaining text from the response tuple
                        //     var remainingText = data[0];
                        //
                        //     // Continue with your code...
                        // } else {
                        //     // Handle the case where data is not a tuple
                        //     console.log('Error: Response is not a tuple');
                        //     var remainingText = data
                        // }

                        // var remainingText = data[0];

                        var botMessage = new Message({
                            // text: marked.parse(data),
                            text: marked.parse(data),
                            message_side: 'left',
                            buttons: [],// No buttons for user message
                        });

                        // Draw bot message
                        botMessage.draw();

                        $messages.animate({scrollTop: $messages.prop('scrollHeight')}, 300);


                    });

                    return $messages.animate({scrollTop: $messages.prop('scrollHeight')}, 300);
                }


                $('.fa-arrow-up').click(function (e) {
                    return sendMessage(getMessageText());
                });

                // Event listener for pressing enter the message input field
                $('.message_input').keyup(function (e) {
                    if (e.which === 13) {
                        return sendMessage(getMessageText());
                    }
                });

                // Add "Writing..." message
                var writingMessage = new Message({
                    text: marked.parse("<p>Hello! I'm CodeGenie </p>" +

                        '\n'
                    ),
                    message_side: 'left',
                    buttons: [ // Add buttons for initial bot message
                        {text: "Write a Python code sample"},
                        {text: "Write code to remove duplicates in Python"},
                        {text: "Generate a barplot"},
                        {text: "Generate a distribution plot"},

                    ]
                });
                writingMessage.draw();

            }
        )
        ;
    }

    ()
)
;
